import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib # For saving preprocessor and label_encoder

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical # For one-hot encoding the labels



df_train = pd.read_csv(r"C:\Users\Arfan\Desktop\meow\data\UNSW_NB15_training-set.csv")
df_test = pd.read_csv(r"C:\Users\Arfan\Desktop\meow\data\UNSW_NB15_testing-set.csv")
print("Datasets loaded successfully.")

# --- 3. Preprocessing (Updated for Multi-Class) ---

# --- 3.1 Combine Data (for consistent preprocessing) ---
df_train['source'] = 'train'
df_test['source'] = 'test'
df_combined = pd.concat([df_train, df_test], ignore_index=True)
print(f"\nCombined data shape: {df_combined.shape}")

# --- 3.2 Define Features (X) and Target (y) ---
# 'id', 'label' are excluded from features. 'attack_cat' is now the target.
cols_to_drop = ['id', 'label', 'source'] # 'source' is temporary
X_combined = df_combined.drop(columns=cols_to_drop)
y_combined_cat = df_combined['attack_cat'] # Multi-class target

# --- 3.3 Identify and Handle Missing Values ---
print("\nMissing values before handling:")
print(X_combined.isnull().sum()[X_combined.isnull().sum() > 0])

# For UNSW-NB15, some 'service' values might be NaN in practice, or other columns might have issues.
if 'service' in X_combined.columns and X_combined['service'].isnull().any():
    print("Handling missing 'service' values by filling with '-'.")
    X_combined['service'].fillna('-', inplace=True)
if 'state' in X_combined.columns and X_combined['state'].isnull().any():
    print("Handling missing 'state' values by filling with '-'.")
    X_combined['state'].fillna('-', inplace=True)

# --- 3.4 & 3.5 Categorical and Numerical Feature Preprocessing ---
categorical_features = X_combined.select_dtypes(include='object').columns
numerical_features = X_combined.select_dtypes(include=np.number).columns

print(f"\nCategorical features identified: {list(categorical_features)}")
print(f"Numerical features identified: {list(numerical_features)}")

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("\nApplying preprocessing steps...")
X_processed_array = preprocessor.fit_transform(X_combined)
print("Preprocessing complete.")
print(f"\nShape of X_processed_array after preprocessing: {X_processed_array.shape}")

# Save the fitted preprocessor
preprocessor_path = 'saved_models/ids_preprocessor.pkl'
joblib.dump(preprocessor, preprocessor_path)
print(f"Preprocessor saved to '{preprocessor_path}'")




# --- 3.6 Encode Multi-Class Target Variable (attack_cat) ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_combined_cat) # Integer encode string categories

# Convert integer labels to one-hot encoded vectors for Keras
y_onehot = to_categorical(y_encoded)

print(f"Original attack categories: {label_encoder.classes_}")
print(f"Shape of one-hot encoded y: {y_onehot.shape}")
num_classes = y_onehot.shape[1] # Number of output classes for the model

# Save the label encoder to decode predictions later
joblib.dump(label_encoder, 'saved_models/label_encoder.pkl')
print("Label encoder saved to 'label_encoder.pkl'")


# --- 3.7 Split back into Training and Testing Sets ---
train_mask = (df_combined['source'] == 'train').values
test_mask = (df_combined['source'] == 'test').values

X_train_processed_array = X_processed_array[train_mask]
y_train_onehot = y_onehot[train_mask] # Use one-hot encoded labels for training

X_test_processed_array = X_processed_array[test_mask]
y_test_onehot = y_onehot[test_mask] # Use one-hot encoded labels for testing


print(f"\nRe-split Training X shape: {X_train_processed_array.shape}, y shape: {y_train_onehot.shape}")
print(f"Re-split Testing X shape: {X_test_processed_array.shape}, y shape: {y_test_onehot.shape}")


# --- 3.8 Reshape Data for 1D CNN Input ---
input_features = X_train_processed_array.shape[1]
input_shape_cnn = (input_features, 1) # (num_features, 1) for 1D CNN

X_train_cnn = X_train_processed_array.reshape(X_train_processed_array.shape[0], input_features, 1)
X_test_cnn = X_test_processed_array.reshape(X_test_processed_array.shape[0], input_features, 1)

print(f"\nCNN Training X shape: {X_train_cnn.shape}")
print(f"CNN Testing X shape: {X_test_cnn.shape}")

# --- 3.9 Handle Class Imbalance (for Training Data) ---
# For multi-class, compute class weights using the integer encoded labels
y_train_labels_for_weights = label_encoder.inverse_transform(np.argmax(y_train_onehot, axis=1))

class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=np.unique(y_train_labels_for_weights),
                                                  y=y_train_labels_for_weights)
class_weights_dict = dict(zip(np.unique(y_train_labels_for_weights), class_weights))

print(f"\nClass distribution in training set (before weights):")
print(pd.Series(y_train_labels_for_weights).value_counts())
print(f"Calculated class weights: {class_weights_dict}")

# --- 4. Build the 1D CNN Model (Updated for Multi-Class) ---
print("\nBuilding 1D CNN model...")
model = Sequential([
    layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape_cnn),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax') # Output layer for multi-class classification
])

# Use 'categorical_crossentropy' for one-hot encoded multi-class labels
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 5. Train the Model ---
print("\nTraining model...")
# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_cnn, y_train_onehot,
                    epochs=50, # You can adjust this
                    batch_size=128, # You can adjust this
                    validation_split=0.2, # Use part of training data for validation
                    class_weight=class_weights_dict, # Apply class weights
                    callbacks=[early_stopping],
                    verbose=1)
print("Model training complete.")
model.save('saved_models/keras_model.h5')
print("Model Saved")