from flask import Flask, request, jsonify
from prometheus_client import make_wsgi_app, Counter, Histogram, Gauge
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import time
import psutil
import random
import threading

app = Flask(__name__)

# =========================================
# 1. DEFINISI METRIK PROMETHEUS
# =========================================
REQUEST_COUNT = Counter('app_request_count', 'Total Request HTTP', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Waktu proses request')
CPU_USAGE = Gauge('system_cpu_usage_percent', 'Penggunaan CPU System')
MEMORY_USAGE = Gauge('system_memory_usage_percent', 'Penggunaan RAM System')

# Middleware agar metrics bisa diakses di /metrics
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

# =========================================
# 2. SIMULASI MONITORING SYSTEM (BACKGROUND)
# =========================================
def monitor_system():
    while True:
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)
        time.sleep(5)

# Jalankan monitoring di thread terpisah
threading.Thread(target=monitor_system, daemon=True).start()

# =========================================
# 3. ENDPOINT PREDIKSI (INFERENCE)
# =========================================
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    # Simulasi processing model
    time.sleep(random.uniform(0.1, 0.5)) 
    
    # Simulasi Prediksi (0: Aman, 1: Berisiko)
    # Kita pakai random dulu agar tidak ribet load model pkl yg besar
    prediction = random.choice([0, 1])
    
    # Catat ke Prometheus
    REQUEST_COUNT.labels(method='POST', endpoint='/predict', status=200).inc()
    REQUEST_LATENCY.observe(time.time() - start_time)
    
    return jsonify({
        'prediction': prediction,
        'status': 'success'
    })

@app.route('/')
def home():
    return "Model Serving is Running! Go to /metrics to see data."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)