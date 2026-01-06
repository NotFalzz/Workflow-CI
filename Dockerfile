FROM python:3.12-slim

WORKDIR /app

# Pastikan requirements.txt ada di folder yang sama saat build
COPY MLProject/requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# PENTING: Kamu butuh file aplikasi inference kamu (Flask/FastAPI)
# Salin file inference kamu ke sini. Namanya apa? (misal: app.py atau Inference.py)
COPY Inference.py . 

# Perintah menjalankan aplikasi
CMD ["python", "Inference.py"]
