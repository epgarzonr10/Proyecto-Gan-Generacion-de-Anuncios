# Generador de Anuncios con GAN

Este proyecto es un generador de anuncios publicitarios que utiliza un modelo Generative Adversarial Network (GAN) para crear fondos personalizados y GPT-Turbo para generar eslóganes. Los usuarios pueden ingresar fotos de productos y logos, y el sistema generará un anuncio con un fondo y eslogan acorde al producto.

## Características

- **Ingreso de Imágenes**: Permite cargar imágenes de productos y logos.
- **Generación de Fondos**: Utiliza un modelo GAN para crear un fondo visual único para el anuncio.
- **Generación de Eslóganes**: Usa GPT-Turbo para generar un eslogan relacionado con el producto y el logo.
- **Interfaz Amigable**: Interfaz de usuario sencilla para facilitar la carga de imágenes y la generación del anuncio.

## Requisitos

- Python 3.8 o superior
- Pytorch (para la implementación del modelo GAN)
- OpenAI GPT API para la generación de eslóganes
- Librerías adicionales: 
  - `numpy`
  - `pillow`
  - `flask` (si se implementa una interfaz web)
  - `requests` (para interactuar con la API de GPT-Turbo)

## Instalación

##1. Clona este repositorio:

git clone https://github.com/tuusuario/generador-anuncios-gan.git

##2. Navega hasta:
cd generador-anuncios-gan

##4. Instala las dependencias:
pip install -r requirements.txt


