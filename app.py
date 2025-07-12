from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('motor_fault_gru_model.h5')
scaler = joblib.load('scaler_motor_fault.pkl')

@app.route('/')
def home():
    return "Motor Fault GRU API is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape((1, 1, 17))
    scaled_input = scaler.transform(input_data.reshape(1, -1)).reshape(1, 1, 17)
    prediction = model.predict(scaled_input)
    return jsonify({'fault_prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)