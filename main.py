import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required

#load TFLite modelnya
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#data untuk label encoding
soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil Seeds', 'Pulses', 'Ground Nuts', 'Rice', 'Pomegranate', 'Coffee', 'Watermelon', 'Kidneybeans', 'Orange']
fertilizer_names = ['10-10-10', '10-26-26', '14-14-14', '14-35-14', '15-15-15', '17-17-17', '20-20', '28-28', 'DAP', 'Potassium chloride', 'Potassium sulfate', 'Superphosphate', 'TSP', 'Urea']

label_encoder_soil = LabelEncoder()
label_encoder_crop = LabelEncoder()
fertilizer_encoder = LabelEncoder()

label_encoder_soil.fit(soil_types)
label_encoder_crop.fit(crop_types)
fertilizer_encoder.fit(fertilizer_names)

onehot_encoder_soil = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
onehot_encoder_crop = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

onehot_encoder_soil.fit(label_encoder_soil.transform(soil_types).reshape(-1, 1))
onehot_encoder_crop.fit(label_encoder_crop.transform(crop_types).reshape(-1, 1))

def preprocess_input(N, P, K, temperature, humidity, soil_type, crop_type):
    soil_encoded = label_encoder_soil.transform([soil_type])
    crop_encoded = label_encoder_crop.transform([crop_type])
    soil_onehot = onehot_encoder_soil.transform(soil_encoded.reshape(-1, 1))
    crop_onehot = onehot_encoder_crop.transform(crop_encoded.reshape(-1, 1))
    input_data = np.hstack([N, P, K, temperature, humidity, soil_onehot.flatten(), crop_onehot.flatten()])
    return input_data.reshape(1, -1)

def predict_fertilizer(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(prediction, axis=1)[0]
    fertilizer_name = fertilizer_encoder.inverse_transform([predicted_class])
    return fertilizer_name[0]

app = Flask(__name__)

#konfigurasi buat JWT tokennya
app.config['JWT_SECRET_KEY'] = 'super-secret'
jwt = JWTManager(app)

@app.route("/token", methods=["POST"])
def get_token():
    token = create_access_token(identity="system_user")
    return jsonify(access_token=token)

@app.route("/kalkulator-pupuk", methods=["POST"])
@jwt_required() 
def index():
    input_data = request.json
    try:
        #ambil data input dari user dengan json
        N = input_data.get('N')
        P = input_data.get('P')
        K = input_data.get('K')
        temperature = input_data.get('temperature')
        humidity = input_data.get('humidity')
        soil_type = input_data.get('soil_type')
        crop_type = input_data.get('crop_type')
    except Exception as e:
        return jsonify({"error": f"Missing input data: {str(e)}"}), 400
    
    try:
        tensor_input = preprocess_input(N, P, K, temperature, humidity, soil_type, crop_type)
        prediction = predict_fertilizer(tensor_input)
        return jsonify({"predicted_fertilizer": prediction})
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)