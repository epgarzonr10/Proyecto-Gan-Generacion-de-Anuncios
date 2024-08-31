import os
import pandas as pd
from PIL import Image
import numpy as np
import albumentations as A

GENERATE_SQUARE = 470
IMAGE_CHANNELS = 3

def augment_image(image):
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.GaussNoise(p=0.5),
        A.ElasticTransform(p=0.5),
    ])
    return transform(image=image)['image']

def preprocess_images(csv_file, output_file):
    # Intentar diferentes codificaciones
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_file, encoding=encoding, sep=',', header=0)
            print(f"Archivo CSV leído exitosamente con codificación {encoding}")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError("No se pudo leer el archivo CSV con ninguna de las codificaciones probadas")

    # Verificar si las columnas están correctamente separadas
    if df.shape[1] == 1:
        df = df.iloc[:, 0].str.split(',', expand=True)
    
    # Imprimir los nombres de las columnas para verificación
    print(f"Columnas del DataFrame: {df.columns}")
    print(f"Forma del DataFrame: {df.shape}")
    
    # Si hay más de 3 columnas, eliminar las adicionales
    if df.shape[1] > 3:
        df = df.iloc[:, :3]
    
    df.columns = ['image_path', 'label', 'label_id']

    processed_images = []
    labels = []

    for _, row in df.iterrows():
        img_path = row['image_path']
        label = row['label_id']

        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.LANCZOS)
            img_array = np.array(img)

            # Aplicar aumento de datos
            augmented_image = augment_image(img_array)
            
            processed_images.append(augmented_image)
            labels.append(label)
            
            # Añadir la imagen original también
            processed_images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error procesando imagen {img_path}: {str(e)}")

    processed_images = np.array(processed_images).astype(np.float32) / 127.5 - 1
    labels = np.array(labels)

    # Construir la ruta para guardar el archivo .npz en la misma carpeta que el archivo CSV
    output_file_path = os.path.join(os.path.dirname(csv_file), output_file)
    
    np.savez(output_file_path, images=processed_images, labels=labels)
    print(f"Datos procesados guardados en {output_file_path}")
    print(f"Forma final de los datos: imágenes {processed_images.shape}, etiquetas {labels.shape}")

if __name__ == "__main__":
    preprocess_images("dataset/image_labels.csv", "processed_images.npz")
