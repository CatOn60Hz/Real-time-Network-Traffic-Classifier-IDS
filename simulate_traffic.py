from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import traceback
import os

# --- Configuration ---
MODEL_DIR = 'saved_models'
MODEL_PATH = os.path.join(MODEL_DIR, 'keras_model.h5')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'ids_preprocessor.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'original_feature_names.pkl')

# --- Load Artifacts ---
print("Loading model and preprocessors...")
try:
    model = load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    original_feature_names = joblib.load(FEATURES_PATH)
    print("Artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading model or preprocessors: {e}")
    traceback.print_exc()
    exit()

# --- Flask App Setup ---
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "IDS Multi-Class Prediction API is running."})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input and convert to DataFrame
        input_data = request.get_json()
        df = pd.DataFrame([input_data])

        # Drop any label-related columns safely
        df = df.drop(columns=['attack_cat', 'label', 'id'], errors='ignore')

        # Ensure all expected features exist and are in order
        missing_features = set(original_feature_names) - set(df.columns)
        if missing_features:
            return jsonify({
                "error": f"Missing input features: {missing_features}",
                "message": "Ensure your input contains all required features."
            }), 400

        df = df[original_feature_names]  # Reorder correctly

        # Preprocess input
        X_processed = preprocessor.transform(df)
        input_features = X_processed.shape[1]
        X_reshaped = X_processed.reshape(1, input_features, 1)  # CNN expects (batch, features, 1)

        # Predict
        preds = model.predict(X_reshaped)
        predicted_index = preds.argmax(axis=1)[0]
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        return jsonify({
            "predicted_attack_category": predicted_label,
            "confidence_scores": preds.tolist()[0]
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Failed to process input data. Ensure correct format and features."
        }), 400

# --- Run Server ---
if __name__ == '__main__':
    app.run(debug=True)
