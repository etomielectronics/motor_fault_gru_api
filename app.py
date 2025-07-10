from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model('motor_fault_gru_model.h5')
scaler = joblib.load('scaler_gru.pkl')  # Use your saved scaler

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_features = [
            data["frequency_red"], data["frequency_blue"], data["frequency_yellow"],
            data["voltage_red_phase"], data["voltage_blue_phase"], data["voltage_yellow_phase"],
            data["current_red_phase"], data["current_blue_phase"], data["current_yellow_phase"],
            data["ambient_temperature"], data["temperature"],
            data["power_factor_red"], data["power_factor_blue"], data["power_factor_yellow"],
            data["vibration_x"], data["vibration_y"], data["vibration_z"]
        ]
        
        input_np = np.array(input_features).reshape(1, 1, -1)  # shape: (1, 1, 17)
        scaled_input = scaler.transform(input_np.reshape(1, -1)).reshape(1, 1, -1)

        prediction = model.predict(scaled_input)
        predicted_class = int(prediction[0][0] > 0.5)  # Assuming binary classification
        label = "Fault" if predicted_class == 1 else "Normal"

        return jsonify({
            "prediction": label,
            "probability": float(prediction[0][0])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
