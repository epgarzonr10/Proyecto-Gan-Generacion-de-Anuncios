from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import os
import glob
import hashlib

def download_images(query, num_images):
    crawler = GoogleImageCrawler(storage={'root_dir': './downloads/raw'})
    crawler.crawl(keyword=query, max_num=num_images)

def hash_image(image_path):
    hasher = hashlib.md5()
    with open(image_path, 'rb') as img_file:
        buf = img_file.read()
        hasher.update(buf)
    return hasher.hexdigest()

def remove_duplicates(source_folder):
    seen_hashes = set()
    for image_path in glob.glob(f"{source_folder}/*"):
        img_hash = hash_image(image_path)
        if img_hash in seen_hashes:
            os.remove(image_path)
        else:
            seen_hashes.add(img_hash)

def resize_and_rename_images(source_folder, destination_folder, size):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    image_number = 1
    for image_path in sorted(glob.glob(f"{source_folder}/*")):
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = img.resize(size, Image.Resampling.LANCZOS)
                file_name = f"montana_{image_number}.png"
                img.save(os.path.join(destination_folder, file_name), "PNG")
                image_number += 1
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# Configuración
query = "montañas"
num_images = 150
source_folder = "./downloads/raw"
destination_folder = "./downloads/renamed"
size = (470, 470)

# Ejecución
download_images(query, num_images)
remove_duplicates(source_folder)
resize_and_rename_images(source_folder, destination_folder, size)