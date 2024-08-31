import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Para reducir los mensajes de logging de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desactivar optimizaciones de oneDNN
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Desactivar algunas optimizaciones de grafo

import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, LeakyReLU, UpSampling2D, Conv2D, Embedding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import Constraint
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import albumentations as A

# Configuraciones
GENERATE_SQUARE = 256  # Tamaño de la imagen ajustado
IMAGE_CHANNELS = 3
SEED_SIZE = 100
EPOCHS = 1000
BATCH_SIZE = 64  # Tamaño del lote ajustado
SAVE_INTERVAL = 100
BUFFER_SIZE = 1000
DATA_PATH = 'dataset'
CHECKPOINT_PATH = 'ad-gen'
NUM_CLASSES = 5

def load_data_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"El archivo {csv_path} no existe.")
    data = pd.read_csv(csv_path)
    images, labels = [], []
    for _, row in data.iterrows():
        image_path = row['image_path']
        label = row['label_id']
        if os.path.exists(image_path):
            image = Image.open(image_path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.LANCZOS)
            images.append(np.asarray(image))
            labels.append(label)
        else:
            print(f"Advertencia: La imagen {image_path} no se encontró y será omitida.")
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    images = (images / 127.5) - 1.0  # Normalizar imágenes
    print(f"Loaded data shape: images {images.shape}, labels {labels.shape}")
    return images, labels

# Cargar datos
csv_path = os.path.join(DATA_PATH, 'image_labels.csv')
images, labels = load_data_from_csv(csv_path)
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Funciones auxiliares
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def augment_image(image):
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.GaussNoise(p=0.5),
        A.ElasticTransform(p=0.5),
        A.Resize(GENERATE_SQUARE, GENERATE_SQUARE),
    ])
    return transform(image=image)['image']

class SpectralNormalization(Constraint):
    def _call_(self, w):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])
        sigma = tf.linalg.norm(w, ord=2)
        w = w / sigma
        w = tf.reshape(w, w_shape)
        return w

# Construcción del generador
def build_generator(seed_size, channels, num_classes):
    noise = Input(shape=(seed_size,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Embedding(num_classes, seed_size)(label)
    label_embedding = Flatten()(label_embedding)
    x = Concatenate()([noise, label_embedding])
    x = Dense(16*16*1024)(x)  # Aumentar la capacidad del generador
    x = Reshape((16, 16, 1024))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D()(x)
    x = Conv2D(512, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D()(x)
    x = Conv2D(256, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D()(x)
    x = Conv2D(128, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = UpSampling2D()(x)
    x = Conv2D(channels, kernel_size=3, padding="same", activation="tanh")(x)
    return Model([noise, label], x)

# Construcción del discriminador
def build_discriminator(image_shape, num_classes):
    image = Input(shape=image_shape)
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Embedding(num_classes, np.prod(image_shape))(label)
    label_embedding = Reshape(image_shape)(label_embedding)
    x = Concatenate()([image, label_embedding])
    x = Conv2D(64, kernel_size=3, strides=2, padding="same", kernel_constraint=SpectralNormalization())(x)
    x = LeakyReLU()(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding="same", kernel_constraint=SpectralNormalization())(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.25)(x)
    x = Conv2D(256, kernel_size=3, strides=2, padding="same", kernel_constraint=SpectralNormalization())(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.25)(x)
    x = Conv2D(512, kernel_size=3, strides=2, padding="same", kernel_constraint=SpectralNormalization())(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model([image, label], x)

# Inicialización de generador y discriminador
image_shape = (GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS)
generator = build_generator(SEED_SIZE, IMAGE_CHANNELS, NUM_CLASSES)
discriminator = build_discriminator(image_shape, NUM_CLASSES)

# Cargar el último checkpoint
initial_epoch = 0
checkpoint_dir = 'checkpoints'
checkpoint_path = os.path.join(checkpoint_dir, f'generator_epoch_{initial_epoch:04d}.weights.h5')
if os.path.exists(checkpoint_path):
    generator.load_weights(checkpoint_path)
    print(f"Checkpoint cargado: {checkpoint_path}")
else:
    print("Checkpoint no encontrado, comenzando desde la primera época.")

# Funciones de pérdida
cross_entropy = tf.keras.losses.BinaryCrossentropy()
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizadores
generator_optimizer = tf.keras.optimizers.Adam(5e-5, 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(5e-5, 0.5)

@tf.function
def train_step(images, labels):
    noise = tf.random.normal([tf.shape(images)[0], SEED_SIZE])
    for _ in range(2):  # Actualiza el discriminador dos veces por cada actualización del generador
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator([noise, labels], training=True)
            real_output = discriminator([images, labels], training=True)
            fake_output = discriminator([generated_images, labels], training=True)
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input, test_labels):
    predictions = model([test_input, test_labels], training=False)
    output_path = os.path.join(DATA_PATH, 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(min(predictions.shape[0], 5)):
        img = (predictions[i, :, :, :] + 1) / 2.0
        img = (img * 255).numpy().astype(np.uint8)
        im = Image.fromarray(img)
        filename = os.path.join(output_path, f'epoch_{epoch:04d}image{i}.png')
        im.save(filename)
        print(f"Imagen guardada: {filename}")

def plot_metrics(epoch, gen_loss_epoch, disc_loss_epoch, gen_losses, disc_losses):
    gen_losses.append(gen_loss_epoch)
    disc_losses.append(disc_loss_epoch)
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    metrics_path = os.path.join(CHECKPOINT_PATH, f'loss_plot_epoch_{epoch:04d}.png')
    plt.savefig(metrics_path)
    plt.close()
    print(f'Métricas guardadas en: {metrics_path}')

gen_losses = []
disc_losses = []

def train(dataset, epochs):
    for epoch in range(initial_epoch, epochs):
        start = time.time()
        gen_loss_epoch = 0
        disc_loss_epoch = 0
        batches = 0
        print(f"Iniciando entrenamiento de la época {epoch+1}/{epochs}")
        for image_batch, label_batch in dataset:
            try:
                label_batch = tf.cast(label_batch, tf.int32)
                gen_loss, disc_loss = train_step(image_batch, label_batch)
                gen_loss_epoch += gen_loss
                disc_loss_epoch += disc_loss
                batches += 1
            except Exception as e:
                print(f"Error en el batch {batches + 1}: {e}")

        if batches > 0:
            gen_loss_epoch /= batches
            disc_loss_epoch /= batches
        
        generate_and_save_images(generator, epoch + 1, seed, labels_input)
        print(f"Imágenes generadas para la época {epoch + 1}")

        if (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'generator_epoch_{epoch+1:04d}.weights.h5')
            generator.save_weights(checkpoint_path)
            print(f"Checkpoint guardado: {checkpoint_path}")
            plot_metrics(epoch + 1, gen_loss_epoch, disc_loss_epoch, gen_losses, disc_losses)

        print(f'Epoch {epoch+1}/{epochs}, gen_loss={gen_loss_epoch:.4f}, disc_loss={disc_loss_epoch:.4f}, time={time.time()-start:.2f}s')
        print(f'Training time: {hms_string(time.time() - start)}')

    generate_and_save_images(generator, epochs, seed, labels_input)
    print(f"Imágenes generadas después del entrenamiento final")
    final_model_path = os.path.join(CHECKPOINT_PATH, 'generator_final.weights.h5')
    generator.save_weights(final_model_path)
    print(f"Modelo final guardado: {final_model_path}")

seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])
labels_input = tf.random.uniform([BATCH_SIZE], minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

for image_batch, label_batch in dataset.take(1):
    print(f"Primer batch - image_batch shape: {image_batch.shape}, label_batch shape: {label_batch.shape}")

train(dataset, EPOCHS + initial_epoch)
model_json = generator.to_json()
architecture_path = os.path.join(CHECKPOINT_PATH, "generator_architecture.json")
with open(architecture_path, "w") as json_file:
    json_file.write(model_json)
print(f"Entrenamiento completado y modelo guardado en {CHECKPOINT_PATH}.")