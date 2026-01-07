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
DAGSHUB_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "NotFalzz")
DAGSHUB_REPO_NAME = "Eksperimen_SML_Renn"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")

if not MY_TOKEN:
    print("⚠️ Peringatan: Token DagsHub tidak ditemukan di .env. Pastikan Secrets GitHub sudah diset jika ini berjalan di CI/CD.")
else:
    print("✅ Token DagsHub ditemukan!")

print(f"Menghubungkan ke DagsHub: {DAGSHUB_REPO_NAME}...")

# Autentikasi DagsHub - Tambahkan error handling
try:
    dagshub.auth.add_app_token(MY_TOKEN)
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    print("✅ Koneksi DagsHub berhasil!")
except Exception as e:
    print(f"❌ Error koneksi DagsHub: {e}")
    raise

# Setup MLflow
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Eksperimen_Credit_Risk_Renn")
mlflow.autolog()

# ==========================================
# 2. LOAD DATA
# ==========================================
print("📂 Memuat data...")

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'credit_risk_preprocessing.csv')

print(f"🔍 Mencari dataset di: {file_path}")

if os.path.exists(file_path):
    print(f"✅ Dataset ditemukan!")
    df = pd.read_csv(file_path)
    print(f"📊 Dataset shape: {df.shape}")
else:
    raise FileNotFoundError(f"❌ File dataset tidak ditemukan di: {file_path}")

# Pisahkan Fitur dan Target
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Data split selesai - Train: {len(X_train)}, Test: {len(X_test)}")

# ==========================================
# 3. TRAINING & UPLOAD
# ==========================================
print("🚀 Mulai Training & Tracking MLflow...")

with mlflow.start_run():
    # Model Training
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Akurasi Model: {acc:.4f}")
    
    print(f"\n🎉 SUKSES! Model dan metrik berhasil dikirim ke DagsHub.")
    print(f"🔗 Cek di sini: {TRACKING_URI}")
