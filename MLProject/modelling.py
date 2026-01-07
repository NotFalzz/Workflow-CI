import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import dagshub
import os
from dotenv import load_dotenv

# ==========================================
# 1. KONFIGURASI KEAMANAN & DAGSHUB
# ==========================================
load_dotenv() 

# Ambil token dari Environment Variable
MY_TOKEN = os.getenv("DAGSHUB_TOKEN")
DAGSHUB_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "NotFalzz") # Fallback ke default jika env kosong
DAGSHUB_REPO_NAME = "Eksperimen_SML_Renn"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")

if not MY_TOKEN:
    print("Peringatan: Token DagsHub tidak ditemukan di .env. Pastikan Secrets GitHub sudah diset jika ini berjalan di CI/CD.")

print(f"Menghubungkan ke DagsHub: {DAGSHUB_REPO_NAME}...")

# Autentikasi DagsHub
dagshub.auth.add_app_token(MY_TOKEN)
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)

# Setup MLflow
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Eksperimen_Credit_Risk_Renn")
# PENTING: autolog diletakkan sebelum training dimulai
mlflow.autolog()

# ==========================================
# 2. LOAD DATA (ANTI-ERROR PATH - REVISI)
# ==========================================
print("Memuat data...")

# Teknik ini memastikan Python mencari file di folder yang SAMA dengan script ini
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'credit_risk_preprocessing.csv')

if os.path.exists(file_path):
    print(f"Dataset ditemukan di: {file_path}")
    df = pd.read_csv(file_path)
else:
    # Error handling yang jelas
    raise FileNotFoundError(f"File dataset tidak ditemukan di: {file_path}. Pastikan file 'credit_risk_preprocessing.csv' ada di dalam folder MLProject!")

# Pisahkan Fitur dan Target
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. TRAINING & UPLOAD
# ==========================================
print("Mulai Training & Tracking MLflow...")

with mlflow.start_run():
    # Model Training
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Akurasi Model: {acc:.4f}")
    
    # Catatan: Karena sudah pakai mlflow.autolog(), kita TIDAK PERLU lagi
    # mengetik mlflow.log_param atau log_model secara manual. 
    # Autolog sudah merekam semuanya otomatis.
    
    print("\n🚀 SUKSES! Model dan metrik berhasil dikirim ke DagsHub.")
    print(f"Cek di sini: {TRACKING_URI}")
