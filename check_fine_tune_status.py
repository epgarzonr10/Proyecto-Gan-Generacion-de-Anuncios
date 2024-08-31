import os
import openai
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configurar la API key de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def list_fine_tune_jobs():
    response = openai.FineTuningJob.list()
    return response['data']

def check_fine_tune_status(job_id):
    response = openai.FineTuningJob.retrieve(job_id)
    return response

# Listar todos los trabajos de fine-tuning
fine_tune_jobs = list_fine_tune_jobs()

# Imprimir los detalles de cada trabajo de fine-tuning
for job in fine_tune_jobs:
    print(f"ID: {job['id']}, Status: {job['status']}, Created At: {job['created_at']}, Model: {job['model']}")

# ID del trabajo de fine-tuning que quieres verificar
#fine_tune_job_id = 'ftjob-e0L9qW9X0wViFb7oef2OOZPk'  # Reemplaza esto con el ID de tu trabajo

# Verificar el estado de un trabajo específico
#fine_tune_status = check_fine_tune_status(fine_tune_job_id)
#print(fine_tune_status)

# ID del trabajo de fine-tuning que quieres cancelar
# fine_tune_job_id = 'ftjob-7E9K7PRmCaYY4mwhwQvDkWCL'  # Reemplaza esto con el ID de tu trabajo

# Cancelar el trabajo de fine-tuning
#cancel_response = openai.FineTuningJob.cancel(fine_tune_job_id)
#print(f"Trabajo de fine-tuning cancelado: {cancel_response}")

# ID del archivo de entrenamiento que deseas eliminar
# training_file_id = 'file-UbWTT7AaCaqPqU9lHJlZ3YSP'  # Reemplaza esto con el ID de tu archivo

# Eliminar el archivo de entrenamiento
# delete_response = openai.File.delete(training_file_id)
# print(f"Archivo de entrenamiento eliminado: {delete_response}")
