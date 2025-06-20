# Multi-Class Network Intrusion Detection System (IDS) with 1D Convolutional Neural Networks (CNN)

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [1. Train and Save the Model](#1-train-and-save-the-model)
  - [2. Run the Flask API](#2-run-the-flask-api)
  - [3. Simulate Network Traffic](#3-simulate-network-traffic)
  - [4. (Optional) Access the Basic Web UI](#4-optional-access-the-basic-web-ui)
- [Results and Performance](#results-and-performance)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

This project develops and deploys a robust, multi-class Network Intrusion Detection System (IDS) capable of identifying various attack types and normal network traffic. Leveraging a 1D Convolutional Neural Network (CNN) architecture, the system is trained on the comprehensive UNSW-NB15 dataset, which features a wide range of modern attacks.

The project demonstrates a complete machine learning workflow, from data preprocessing and model training to rigorous evaluation and API deployment, showcasing practical MLOps skills for network security applications.

---

## Key Features

- **Multi-Class Classification:** Accurately distinguishes between 10 distinct categories of network traffic (9 attack types and 1 normal).
- **Deep Learning (1D CNN):** Utilizes a 1D CNN for efficient feature extraction and pattern recognition from sequential network flow data, achieving high detection accuracy.
- **End-to-End MLOps Pipeline:** Demonstrates a complete machine learning workflow, including data preprocessing (scaling, one-hot encoding), model training, rigorous evaluation (with a classification report for each attack type), and API deployment.
- **Scalable Deployment:** Implemented as a Flask API, allowing for real-time inference of incoming network flow data, simulating a practical deployment scenario.
- **High Performance:** Achieves high accuracy (e.g., over 99.9% as reported) on the test set, with strong precision and recall across most attack categories.

---

## Technologies Used

- **Python 3.x**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Flask**
- **Requests**
- **Joblib**

---

## Dataset

This project utilizes the **UNSW-NB15 Dataset**, a publicly available dataset specifically designed for evaluating Intrusion Detection Systems.

### Download Instructions

1. Download both `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`.
2. Create a folder named `data` in your project root directory.
3. Place both CSV files inside the `data/` folder.

your_project_root/
├── data/
│ ├── UNSW_NB15_training-set.csv
│ └── UNSW_NB15_testing-set.csv

---

## Project Structure

your_project_root/
├── app.py # Flask API for model inference and simple UI
├── train_and_save_model.py # Script to preprocess data, train model, and save artifacts
├── simulate_traffic.py # Script to simulate network traffic and send to API
├── data/
│ ├── UNSW_NB15_training-set.csv
│ └── UNSW_NB15_testing-set.csv
├── saved_models/
│ ├── cnn_ids_model.h5
│ ├── ids_preprocessor.pkl
│ ├── label_encoder.pkl
│ └── original_feature_names.pkl
├── templates/
│ └── index.html # Optional basic web UI
├── .gitignore
└── README.md


## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/CatOn60Hz/Real-time-Network-Traffic-Classifier-IDS.git
cd Real-time-Network-Traffic-Classifier-IDS
2. Create and Activate Virtual Environment
bash
Copy
Edit
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
3. Install Dependencies
bash
Copy
Edit
pip install pandas numpy scikit-learn tensorflow flask requests joblib
4. Prepare Dataset
Place the CSV files in the data/ folder as described in the Dataset section.

Usage
1. Train and Save the Model
bash
Copy
Edit
python train_and_save_model.py
This will:

Preprocess the data

Train the 1D CNN

Save:

cnn_ids_model.h5

ids_preprocessor.pkl

label_encoder.pkl

original_feature_names.pkl

to the saved_models/ directory.

2. Run the Flask API
bash
Copy
Edit
python app.py
Starts a Flask server at: http://127.0.0.1:5000/

Keep this terminal running to accept predictions.

3. Simulate Network Traffic
bash
Copy
Edit
# In a new terminal (activate the virtualenv again if needed)
python simulate_traffic.py
Random samples from the test dataset will be sent to the /predict API.

