# app.py
from flask import Flask, request, jsonify, render_template # Add render_template if you plan to use UI
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os
import warnings

# Suppress warnings from scikit-learn when loading models/preprocessors
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

app = Flask(__name__)

# --- Configuration ---
MODEL_SAVE_DIR = 'saved_models/'
MODEL_NAME = 'keras_model.h5' # Ensure this matches what you saved (e.g., cnn_ids_model.h5 or cnn_ids_model_tf_format)
# FIX HERE: Corrected preprocessor filename
PREPROCESSOR_NAME = 'ids_preprocessor.pkl'
# ADD HERE: Filename for your original feature names list
ORIGINAL_FEATURES_LIST_NAME = 'original_feature_names.pkl'
# ADD HERE: Filename for your label encoder (for multi-class output)
LABEL_ENCODER_NAME = 'label_encoder.pkl'


# --- Global variables for loaded model and preprocessor ---
model = None
preprocessor = None # This will be the ColumnTransformer
original_features = None # To store the ordered list of features from training
label_encoder = None # For decoding multi-class predictions


def load_artifacts():
    """Loads the trained model and preprocessor on application startup."""
    global model, preprocessor, original_features, label_encoder

    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    preprocessor_path = os.path.join(MODEL_SAVE_DIR, PREPROCESSOR_NAME)
    original_features_list_path = os.path.join(MODEL_SAVE_DIR, ORIGINAL_FEATURES_LIST_NAME)
    label_encoder_path = os.path.join(MODEL_SAVE_DIR, LABEL_ENCODER_NAME)


    try:
        # Load the model
        if MODEL_NAME.endswith('.h5'):
            model = load_model(model_path)
        else: # Assuming it's a TF SavedModel directory
            model = load_model(model_path) # Keras load_model handles SavedModel directories
        print(f"Model loaded successfully from {model_path}")

        # Load the ColumnTransformer (the main preprocessor for features)
        preprocessor = joblib.load(preprocessor_path)
        print(f"Preprocessor loaded successfully from {preprocessor_path}")

        # Load the original feature names list
        original_features = joblib.load(original_features_list_path)
        print(f"Original features list loaded successfully from {original_features_list_path}")
        print(f"Derived original_features count: {len(original_features)}")

        # Load the LabelEncoder (for decoding multi-class predictions)
        label_encoder = joblib.load(label_encoder_path)
        print(f"Label Encoder loaded successfully from {label_encoder_path}")


    except FileNotFoundError as fnfe:
        print(f"Error: Missing file. Please ensure all saved model artifacts are in '{MODEL_SAVE_DIR}'.")
        print(f"Missing: {fnfe}")
        model = None
        preprocessor = None
        original_features = None
        label_encoder = None
    except Exception as e:
        print(f"Error loading model or preprocessor: {e}")
        import traceback
        traceback.print_exc()
        model = None
        preprocessor = None
        original_features = None
        label_encoder = None

# Load artifacts when the app starts
with app.app_context():
    load_artifacts()

# --- New Route to serve the UI (if you have index.html in 'templates' folder) ---
@app.route('/')
def index():
    # This will look for 'index.html' inside the 'templates' folder
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None or original_features is None or label_encoder is None:
        return jsonify({"error": "Model, preprocessor, or label encoder not loaded. Check server logs."}), 500

    data = request.get_json(force=True)
    # data is expected to be a dictionary representing a single network flow
    # e.g., {"proto": "tcp", "service": "http", "spkts": 10, ...}

    # Convert input data to DataFrame
    try:
        input_df = pd.DataFrame([data])

        # Ensure all expected features are present and in the correct order
        # Fill missing numerical features with 0, and categorical with '-' (or a suitable default)
        # based on how your preprocessor handles unknowns.
        for col in original_features:
            if col not in input_df.columns:
                # This logic should mirror how you filled NaNs during preprocessing (e.g., service and state)
                # For numerical, 0 is often safe. For categorical, '-' might be appropriate if trained with it.
                if col in preprocessor.named_transformers_['cat'].named_steps['onehot'].feature_names_in_:
                    input_df[col] = '-' # Assuming '-' was used for missing categoricals
                else:
                    input_df[col] = 0 # Default for numerical
            # Ensure the data type for numerical columns is consistent if it's coming from an untyped source
            # This can prevent subtle errors during scaling if the number is read as a string.
            if col in preprocessor.named_transformers_['num'].named_steps['scaler'].feature_names_in_:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0) # Convert to numeric, handle potential NaNs from coerce

        # Reorder columns to match the training order BEFORE preprocessing
        input_df = input_df[original_features]

        # Preprocess features
        processed_features = preprocessor.transform(input_df).astype(np.float32)

        # Reshape for CNN input (1 sample, num_features, 1)
        input_features_count = processed_features.shape[1]
        processed_features_cnn = processed_features.reshape(1, input_features_count, 1)

        # Make prediction (multi-class)
        prediction_probs = model.predict(processed_features_cnn)[0] # Get probabilities for all classes
        predicted_class_index = np.argmax(prediction_probs) # Get the index of the highest probability
        predicted_category = label_encoder.inverse_transform([predicted_class_index])[0] # Decode to string name

        # Get the probability for the predicted class
        prediction_probability_for_class = prediction_probs[predicted_class_index]


        response = {
            'predicted_category': predicted_category,
            'prediction_probability_for_category': float(prediction_probability_for_class),
            'all_class_probabilities': {label_encoder.classes_[i]: float(prediction_probs[i]) for i in range(len(label_encoder.classes_))}
        }
        return jsonify(response)
    except Exception as e:
        print(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "message": "Failed to process input data. Ensure correct format and features."}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)