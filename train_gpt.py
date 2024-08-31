import os
import openai
from dotenv import load_dotenv
import pandas as pd
import json
import tenacity

# Cargar variables de entorno
load_dotenv()

# Configurar la API key de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Obtén la ruta absoluta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construye las rutas absolutas
data_dir = os.path.join(current_dir, 'data')
all_slogans_path = os.path.join(data_dir, 'all_slogans.jsonl')

def prepare_slogan_data():
    all_slogans_csv = os.path.join(data_dir, 'all_slogans.csv')
    if os.path.exists(all_slogans_path) and os.path.getsize(all_slogans_path) > 0:
        print(f"El archivo {all_slogans_path} ya existe y no está vacío. Usando el archivo existente.")
        return
    
    print(f"Leyendo slogans desde: {all_slogans_csv}")
    
    try:
        df = pd.read_csv(all_slogans_csv, on_bad_lines='skip', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(all_slogans_csv, on_bad_lines='skip', encoding='latin1')
    except pd.errors.ParserError as e:
        print(f"Error al leer el archivo CSV: {e}")
        return
    
    chat_data = []

    for _, row in df.iterrows():
        category = row['category'] if 'category' in df.columns else 'uncategorized'
        slogan = row['slogan'] if 'slogan' in df.columns else row.iloc[0]
        chat_data.append({
            "messages": [
                {"role": "system", "content": "Eres un generador de slogans."},
                {"role": "user", "content": f"Genera un slogan para la categoría {category}."},
                {"role": "assistant", "content": slogan}
            ]
        })

    print(f"Total de eslóganes únicos encontrados: {len(chat_data)}")
    
    with open(all_slogans_path, 'w', encoding='utf-8') as f:
        for data in chat_data:
            json_line = json.dumps(data)
            f.write(json_line + '\n')
    
    print(f"Eslóganes únicos guardados en: {all_slogans_path}")

def cancel_pending_jobs():
    jobs = openai.FineTuningJob.list()["data"]
    for job in jobs:
        if job["status"] in ["pending", "running"]:
            print(f"Cancelando trabajo: {job['id']} con estado: {job['status']}")
            openai.FineTuningJob.cancel(job["id"])

def fine_tune_model():
    prepare_slogan_data()
    
    print("Iniciando fine-tuning del modelo...")
    
    # Subir el archivo de entrenamiento
    upload_response = openai.File.create(
        file=open(all_slogans_path, "rb"),
        purpose='fine-tune'
    )
    file_id = upload_response['id']
    print(f"Archivo subido con ID: {file_id}")
    
    cancel_pending_jobs()  # Cancelar trabajos pendientes antes de iniciar uno nuevo
    
    # Iniciar el trabajo de fine-tuning
    fine_tune_response = openai.FineTuningJob.create(
        training_file=file_id,
        model="gpt-3.5-turbo-0613"  # Especifica el modelo que deseas afinar
    )
    
    print(f"Fine-tuning iniciado. ID del trabajo: {fine_tune_response['id']}")
    print("El proceso puede tardar varias horas. Puedes verificar el estado con openai.FineTuningJob.retrieve(id)")

if __name__ == "__main__":
    fine_tune_model()
