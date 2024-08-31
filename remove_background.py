# remove_background.py
from rembg import remove
from PIL import Image
import numpy as np

def remove_background(image):
    # Convertir la imagen de PIL a un arreglo numpy
    input_image = np.array(image)

    # Remover el fondo
    output = remove(input_image)

    # Convertir la imagen resultante de nuevo a una imagen PIL
    output_image = Image.fromarray(output, 'RGBA')

    return output_image