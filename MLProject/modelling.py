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
# Muat variabel lingkungan dari file .env
load_dotenv() 

# Ambil token dari .env (Lebih aman!)
MY_TOKEN = os.getenv("DAGSHUB_TOKEN")

if not MY_TOKEN:
    raise ValueError("Token DagsHub tidak ditemukan! Pastikan file .env sudah diisi.")

# Konfigurasi Akun
DAGSHUB_USERNAME = "NotFalzz"
DAGSHUB_REPO_NAME = "Eksperimen_SML_Renn"
TRACKING_URI = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"

print(f"Menghubungkan ke DagsHub: {DAGSHUB_REPO_NAME}...")

# Autentikasi DagsHub
dagshub.auth.add_app_token(MY_TOKEN)
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)


# Setup MLflow
mlflow.autolog()
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Eksperimen_Credit_Risk_Renn")

# ==========================================
# 2. LOAD DATA (ANTI-ERROR PATH)
# ==========================================
print("Memuat data...")

# Cek dua kemungkinan lokasi file (biar dijalankan dari root atau folder tetap jalan)
path_1 = 'Membangun_model/credit_risk_preprocessing/credit_risk_clean.csv'
path_2 = 'credit_risk_preprocessing/credit_risk_clean.csv'

if os.path.exists(path_1):
    df = pd.read_csv(path_1)
elif os.path.exists(path_2):
    df = pd.read_csv(path_2)
else:
    raise FileNotFoundError("File dataset tidak ditemukan di kedua lokasi path!")

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
    
    # Log Metrics & Params
    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", acc)
    
    # Log Model (Simpan model ke DagsHub)
    mlflow.sklearn.log_model(model, "model_random_forest")
    
    print("\n🚀 SUKSES! Model dan metrik berhasil dikirim ke DagsHub.")
    print(f"Cek di sini: {TRACKING_URI}")