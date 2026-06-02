from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sys
import os
import json

# Добавляем пути для импорта ядра
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../m1_v11')))

app = Flask(__name__)
CORS(app)

# Хранилище для последних метрик (упрощенно)
latest_metrics = {
    "layers": {
        "m1": 0, "0": 0, "p1": 0, "p2": 0, "p3": 0, "p4": 0
    },
    "energy": 0,
    "step": 0
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metrics')
def get_metrics():
    # В реальной системе здесь будет чтение из общей памяти или сокета ядра
    # Для теста вернем текущие значения
    return jsonify(latest_metrics)

@app.route('/api/schedule', methods=['POST'])
def schedule_task():
    data = request.json
    # Логика передачи задачи в scheduler.py
    with open('Learning/tasks.json', 'w') as f:
        json.dump(data, f)
    return jsonify({"status": "success", "task": data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
