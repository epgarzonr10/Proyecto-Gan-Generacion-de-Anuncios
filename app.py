import streamlit as st  # Framework para crear aplicaciones web interactivas
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageChops  # Librerías para trabajar con imágenes y aplicar filtros
import os  # Librería para interactuar con el sistema operativo
import openai  # Librería para interactuar con la API de OpenAI
from dotenv import load_dotenv  # Librería para cargar variables de entorno desde un archivo .env
import numpy as np  # Librería para operaciones numéricas y matrices
import tensorflow as tf  # Framework de aprendizaje profundo de Google
from tensorflow.keras.models import model_from_json  # Función para cargar modelos de Keras desde JSON
from remove_background import remove_background  # Función personalizada para eliminar el fondo de las imágenes
import textwrap  # Librería para ajustar texto en líneas
import random  # Librería para generar números aleatorios
from pathlib import Path
import time  # Librería para simular tiempos de espera

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Para reducir los mensajes de logging de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desactivar optimizaciones de oneDNN
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Desactivar algunas optimizaciones de grafo

# Cargar variables de entorno desde un archivo .env
load_dotenv()
tf.config.set_visible_devices([], 'GPU')

# Configurar la API key de OpenAI a partir de las variables de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuraciones generales
GENERATE_SQUARE = 600  # Tamaño de la imagen generada
IMAGE_CHANNELS = 3  # Número de canales de color en la imagen (RGB)
SEED_SIZE = 100  # Tamaño de la semilla para la generación de ruido
NUM_CLASSES = 3  # Número de clases posibles para la clasificación de imágenes

# Rutas de los archivos necesarios
current_dir = os.path.dirname(os.path.abspath(_file_))  # Directorio actual del script
checkpoint_dir = os.path.join(current_dir, 'training_checkpoints')  # Directorio para guardar checkpoints de entrenamiento
model_architecture_path = os.path.join(current_dir, 'ad-gen', 'generator_architecture.json')  # Ruta del archivo JSON con la arquitectura del modelo
model_weights_path = os.path.join(current_dir, 'ad-gen', 'generator_final.weights.h5')  # Ruta del archivo con los pesos del modelo
fonts_dir = os.path.join(current_dir, 'fonts')  # Directorio de fuentes

# ID del modelo fine-tuned (reemplaza esto con el ID real después del fine-tuning)
FINE_TUNED_MODEL = "ftjob-Kq2k2xKL44WKbom0Vl5bcxfgt0MLAu"  # ID del modelo afinado en OpenAI
MODEL = "gpt-3.5-turbo-instruct"

# Función para cargar el modelo GAN
@st.cache(allow_output_mutation=True)
def load_gan_model():
    # Cargar la arquitectura del modelo desde el archivo JSON
    with open(model_architecture_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    generator = model_from_json(loaded_model_json)  # Crear el modelo a partir del JSON
    
    # Cargar los pesos del modelo desde el archivo de pesos
    generator.load_weights(model_weights_path)
    return generator  # Retornar el modelo generador cargado

# Definir las rutas a las texturas para cada categoría
textures = {
    'artisticos': 'textures/artistico/',
    'colores': 'textures/color/',
    'blancos': 'textures/blanco/'
}

def get_random_texture_path(background_type):
    background_type = background_type.lower()
    
    if background_type in ['artisticos', 'blancos', 'colores']:
        texture_dir = Path(textures[background_type])
        texture_files = list(texture_dir.glob('.png')) + list(texture_dir.glob('.jpg'))  # Busca archivos PNG y JPG
        if texture_files:
            return str(random.choice(texture_files))  # Devuelve una textura aleatoria
    
    return textures.get(background_type)


def apply_filter_to_background(image, background_type):
    # Aplicar un filtro de suavizado
    smoothed_image = image.filter(ImageFilter.SMOOTH_MORE)
    
    # Aplicar un filtro de desenfoque gaussiano para suavizar más la imagen
    blurred_image = smoothed_image.filter(ImageFilter.GaussianBlur(radius=4))

    # Aplicar un ajuste de contraste
    enhancer = ImageEnhance.Contrast(blurred_image)
    contrast_image = enhancer.enhance(1.3)  # Ajusta el contraste, 1.0 es el valor original

    # Seleccionar la textura basada en el tipo de fondo
    texture_path = get_random_texture_path(background_type)
    
    # Definir el valor de alpha según el tipo de fondo
    if background_type.lower() == 'artisticos':
        alpha = 0.23
    else:
        alpha = 0.16
    
    # Si se proporciona una textura, aplicarla
    if texture_path:
        texture = Image.open(texture_path).convert('RGB')  # Convertir la textura a modo RGB
        texture = texture.resize(contrast_image.size)  # Redimensionar la textura al tamaño del fondo
        
        # Combinar la textura con la imagen utilizando la mezcla con transparencia
        blended_image = Image.blend(contrast_image, texture, alpha=alpha)  # Ajusta el valor alpha según el tipo de fondo
    
    return blended_image

# Función para contar el número de palabras en un texto, excluyendo caracteres como tildes, espacios, comas o puntos
def count_words(text):
    # Usamos un regex para eliminar caracteres no deseados (comas, puntos, tildes, etc.)
    cleaned_text = ''.join(char for char in text if char.isalnum() or char.isspace())
    words = cleaned_text.split()  # Dividimos el texto en palabras
    return len(words)  # Contamos y retornamos el número de palabras

# Función para convertir un prompt en una etiqueta numérica para la generación de fondo
def convert_prompt_to_label(background_type):
    keywords = {
        'artisticos': 0, 'colores': 1, 'blancos': 2
    }
    return keywords.get(background_type.lower(), random.randint(0, NUM_CLASSES - 1))

# Función para generar un fondo utilizando el modelo GAN y aplicar el filtro
def generate_background(generator, background_type):
    label = convert_prompt_to_label(background_type)
    noise = tf.random.normal([1, SEED_SIZE])
    label_input = tf.constant([label], dtype=tf.int32)
    
    generated_image = generator([noise, label_input], training=False)
    generated_image = generated_image * 0.5 + 0.5
    generated_image = tf.image.resize(generated_image, [GENERATE_SQUARE, GENERATE_SQUARE])
    background = Image.fromarray((generated_image[0] * 255).numpy().astype(np.uint8))
    
    # Aplica el filtro de suavizado, desenfoque, contraste y textura
    filtered_background = apply_filter_to_background(background, background_type)
    
    return filtered_background

# Función para generar un slogan utilizando OpenAI
def generate_slogan(prompt, tone, max_words):
    try:
        slogan = ""
        
        # Continuamos generando eslogans hasta que la cantidad de palabras sea igual a max_words
        while True:
            # Llamar a la API de OpenAI para generar un slogan basado en el prompt, tono y número máximo de palabras
            response = openai.Completion.create(
                model=MODEL,
                prompt=f"Tono: {tone}\n\nGenera un slogan para esta información: {prompt}\n\nSlogan:",
                max_tokens=50,  # Máximo número de tokens en la respuesta
                n=1,  # Número de respuestas a generar
                stop=None,
                temperature=0.7,  # Controla la aleatoriedad de la respuesta
            )
            # Obtenemos el slogan generado
            slogan = response.choices[0].text.strip()
            
            # Contamos el número de palabras
            word_count = count_words(slogan)
            
            # Si el número de palabras es exactamente max_words, rompemos el bucle
            if word_count == max_words:
                break
        
        return slogan  # Retornar el texto del slogan generado
    except Exception as e:
        print(f"Error al generar el slogan: {e}")
        return "No se pudo generar un slogan."  # Retornar mensaje de error en caso de excepción

# Función para crear el anuncio
def create_ad(background, product_image, logo, slogan):
    # Aquí no es necesario convertir background nuevamente a una imagen de PIL.
    ad = background.convert('RGB')
    # Obtener el ancho y alto del anuncio
    ad_width, ad_height = ad.size
    # Redimensionar la imagen del producto basado en un tamaño máximo relativo al anuncio
    max_product_width = ad_width * 0.5  # El producto no debe superar el 50% del ancho del anuncio
    max_product_height = ad_height * 0.5  # El producto no debe superar el 50% de la altura del anuncio

    product_ratio = min(max_product_width / product_image.width, max_product_height / product_image.height)
    
    if product_ratio < 1:
        new_product_size = (int(product_image.width * product_ratio), int(product_image.height * product_ratio))
        product_image = product_image.resize(new_product_size, Image.LANCZOS)

    # Obtener el ancho y alto de la imagen del producto
    product_width, product_height = product_image.size
    # Obtener el ancho y alto del anuncio
    ad_width, ad_height = ad.size
    # Calcular la posición X para centrar la imagen del producto en el anuncio
    x = (ad_width - product_width) // 2
    # Calcular la posición Y para centrar la imagen del producto en el anuncio
    y = (ad_height - product_height) // 2
    # Ajustar la posición Y para subir un poco más la imagen del producto
    offset = 30  # Ajusta este valor según necesites
    y = max(0, y - offset)
    ad.paste(product_image, (x, y), product_image)

    # Redimensionar el logo basado en un tamaño máximo relativo al anuncio
    max_logo_width = ad_width * 0.2  # El logo no debe superar el 20% del ancho del anuncio
    max_logo_height = ad_height * 0.2  # El logo no debe superar el 20% de la altura del anuncio

    logo_ratio = min(max_logo_width / logo.width, max_logo_height / logo.height)
    
    if logo_ratio < 1:
        new_logo_size = (int(logo.width * logo_ratio), int(logo.height * logo_ratio))
        logo = logo.resize(new_logo_size, Image.LANCZOS)
    
    logo = remove_background(logo)  # Eliminar el fondo del logo
    
    # Posición fija en la parte superior derecha
    logo_x = ad_width - logo.width - 20  # Margen derecho de 20 px
    logo_y = 20  # Margen superior de 20 px
    ad.paste(logo, (logo_x, logo_y), logo)

    if slogan:
        draw = ImageDraw.Draw(ad)
        
        # Seleccionar una fuente aleatoria de la carpeta fonts
        font_files = [os.path.join(fonts_dir, font) for font in os.listdir(fonts_dir) if font.endswith(('.otf', '.ttf'))]
        selected_font = random.choice(font_files)
        font = ImageFont.truetype(selected_font, 45)
        
        # Ajustar el slogan al ancho de la imagen
        max_width = ad_width - 10  # Margen de 60 px a cada lado
        avg_char_width = sum(font.getbbox(char)[2] - font.getbbox(char)[0] for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") / len("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        max_chars_per_line = int(max_width / avg_char_width)
        
        wrapped_text = textwrap.fill(slogan, width=max_chars_per_line)
        
        # Dibujar el texto centrado con borde
        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = (ad_width - text_width) // 2
        text_y = ad_height - text_height - 70  # Margen inferior de 70 px
        
        # Dibujar borde alrededor del texto
        outline_color = "black"
        draw.text((text_x - 2, text_y - 2), wrapped_text, font=font, fill=outline_color)
        draw.text((text_x + 2, text_y - 2), wrapped_text, font=font, fill=outline_color)
        draw.text((text_x - 2, text_y + 2), wrapped_text, font=font, fill=outline_color)
        draw.text((text_x + 2, text_y + 2), wrapped_text, font=font, fill=outline_color)
        
        # Dibujar texto principal
        draw.text((text_x, text_y), wrapped_text, font=font, fill=(255, 255, 255))
    
    return ad

# Función para generar un anuncio publicitario completo
def generar_anuncio(prompt, imagen_producto, logo, tone, max_words, background_type):
    # Cargar el modelo GAN para la generación de fondos
    generator = load_gan_model()
    # Generar el fondo del anuncio utilizando el modelo GAN y el tipo de fondo seleccionado
    background = generate_background(generator, background_type)
    # Generar un eslogan basado en el prompt, tono y número máximo de palabras especificado
    slogan = generate_slogan(prompt, tone, max_words)
    # Abrir la imagen del producto y convertirla a formato RGBA (incluye canal alfa para transparencia)
    product_image = Image.open(imagen_producto).convert('RGBA')
    # Remover el fondo de la imagen del producto para obtener solo el producto
    product_image = remove_background(product_image)
    # Abrir el logo y convertirlo a formato RGBA
    logo = Image.open(logo).convert('RGBA')
    # Crear el anuncio combinando el fondo, la imagen del producto, el logo y el eslogan
    ad = create_ad(background, product_image, logo, slogan)
    
    output_path = "generated_ad.png"
    ad.save(output_path)
    
    # Devolver la ruta del archivo generado y el eslogan
    return output_path, slogan

# Función para generar un anuncio publicitario completo
def generar_anuncio(prompt, imagen_producto, logo, tone, max_words, background_type, progress_bar, status_text):
    # Paso 1: Cargar el modelo GAN
    status_text.text("Cargando el modelo GAN...")
    progress_bar.progress(10)
    generator = load_gan_model()
    
    # Paso 2: Generar el fondo del anuncio
    status_text.text("Generando el fondo del anuncio...")
    progress_bar.progress(30)
    background = generate_background(generator, background_type)
    
    # Paso 3: Generar el eslogan
    status_text.text("Generando el eslogan...")
    progress_bar.progress(60)
    slogan = generate_slogan(prompt, tone, max_words)
    
    # Paso 4: Procesar las imágenes del producto y del logo
    status_text.text("Procesando las imágenes del producto y del logo...")
    progress_bar.progress(80)
    product_image = Image.open(imagen_producto).convert('RGBA')
    product_image = remove_background(product_image)
    logo = Image.open(logo).convert('RGBA')
    
    # Paso 5: Crear el anuncio
    status_text.text("Creando el anuncio final...")
    ad = create_ad(background, product_image, logo, slogan)
    
    # Simular un pequeño tiempo de espera para completar la barra de progreso
    time.sleep(1)
    progress_bar.progress(100)
    
    output_path = "generated_ad.png"
    ad.save(output_path)
    
    return output_path, slogan

# Configuración de la página Streamlit
st.set_page_config(page_title="Generador de Anuncios", layout="wide")

st.markdown("<h1 style='text-align: center;'>ANUNCIOS PUBLICITARIOS - GAN</h1>", unsafe_allow_html=True)
st.markdown("""---""")

# Establecer columnas para el front
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    prompt = st.text_area("Detalle toda la información sobre su empresa, producto, servicio (misión, visión):", 
                        "Por ejemplo: Somos una empresa de comida rápida que ofrece hamburguesas jugosas y saludables.")
    tones = ["llamativo", "profesional", "creativo",
             "inspirador", "informal", "juvenil",
             "sofisticado", "enérgico", "minimalista",
             "amigable"]
    selected_tone = st.selectbox("Tono de formalidad para el slogan", tones)
    background_types = ["artisticos", "blancos", "colores"]
    selected_background_type = st.selectbox("Tipo de fondo para el anuncio", background_types)
    max_words = st.slider("Número máximo de palabras para el slogan", min_value=3, max_value=12, value=5)

with col2:
    imagen_producto = st.file_uploader("Suba la imagen de su producto", type=["jpg", "jpeg", "png"])
    st.write("Vista previa del producto:")
    if imagen_producto:
        st.image(Image.open(imagen_producto), width=100)
    logo = st.file_uploader("Suba el logo de su marca", type=["jpg", "jpeg", "png"])
    st.write("Vista previa del logo:")
    if logo:
        st.image(Image.open(logo), width=100)

with col3:
    if st.button("Generar Anuncio"):
        if prompt and imagen_producto and logo:
            # Crear una barra de progreso y un campo de texto para mostrar el estado
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Iniciando generación de anuncio...")
            
            # Generar el anuncio y actualizar la barra de progreso y texto de estado
            resultado, slogan = generar_anuncio(prompt, imagen_producto, logo, selected_tone, max_words, selected_background_type, progress_bar, status_text)
            
            st.success("Anuncio generado con éxito!")
            st.write(f"Slogan generado: {slogan}")
            st.image(resultado, caption="Anuncio generado", width=450, use_column_width=False)
            
            # Añadir botón de descarga de la imagen generada
            with open(resultado, "rb") as file:
                btn = st.download_button(
                    label="Descargar imagen",
                    data=file,
                    file_name="anuncio_generado.png",
                    mime="image/png"
                )
        else:
            st.error("Por favor, complete todos los campos y suba las imágenes necesarias.")