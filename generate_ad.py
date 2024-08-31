# generate_ad.py
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from remove_background import remove_background
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from train_gan import build_generator
import os

def load_gan_model(weights_path):
    model = build_generator(100, 3)  # Ajusta según tu SEED_SIZE e IMAGE_CHANNELS
    model.load_weights(weights_path)
    return model

def generate_background(gan_model, prompt):
    seed = tf.random.normal([1, 100])  # Ajusta según tu SEED_SIZE
    generated_image = gan_model.predict(seed)
    generated_image = (generated_image + 1) / 2.0  # Desnormalizar
    return (generated_image[0] * 255).astype(np.uint8)

def generate_slogan(nlp_model, tokenizer, prompt, category):
    input_text = f"{category}: {prompt}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = nlp_model.generate(input_ids, max_length=50, num_return_sequences=1,
                                no_repeat_ngram_size=2, top_k=50, top_p=0.95,
                                temperature=0.7)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Eliminar la categoría del texto generado
    return generated_text.split(": ", 1)[1] if ": " in generated_text else generated_text

def create_ad(background, product_image, logo, slogan, output_path):
    # Crear una nueva imagen con el fondo generado
    ad = Image.fromarray(background)
    
    # Añadir el producto sin fondo
    product = Image.fromarray(product_image)
    ad.paste(product, (50, 50), product)  # Ajusta la posición según necesites
    
    # Añadir el logo
    logo_img = Image.open(logo)
    ad.paste(logo_img, (400, 400), logo_img)  # Ajusta la posición según necesites
    
    # Añadir el slogan
    draw = ImageDraw.Draw(ad)
    # Usar una fuente por defecto si no se encuentra la especificada
    try:
        font = ImageFont.truetype("arial.ttf", 36)  # Cambia "arial.ttf" por una fuente que tengas instalada
    except IOError:
        font = ImageFont.load_default()
    
    draw.text((10, 10), slogan, font=font, fill=(255, 255, 255))
    
    # Guardar el anuncio final
    ad.save(output_path)

def generate_ad(prompt, product_image_path, logo_path, output_path, category):
    # Cargar modelos
    gan_model = load_gan_model('ad-gen/background_generator_weights.weights.h5')
    nlp_model = GPT2LMHeadModel.from_pretrained('nlp_model')
    tokenizer = GPT2Tokenizer.from_pretrained('nlp_model')
    
    # Generar fondo
    background = generate_background(gan_model, prompt)
    
    # Generar slogan
    slogan = generate_slogan(nlp_model, tokenizer, prompt, category)
    
    # Remover fondo del producto
    product_without_bg = remove_background(product_image_path)
    
    # Crear el anuncio
    create_ad(background, product_without_bg, logo_path, slogan, output_path)

# Ejemplo de uso:
# generate_ad("Hamburguesa jugosa y deliciosa", "path/to/hamburger.jpg", "path/to/logo.png", "path/to/output_ad.png", "food")