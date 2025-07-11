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

<pre> <code>
your_project_root/
├── data/
│ ├── UNSW_NB15_training-set.csv
│ └── UNSW_NB15_testing-set.csv
</code></pre>

---

## Project Structure

<pre> <code>
your_project_root/
├── app.py 
├── train_and_save_model.py 
├── simulate_traffic.py 
├── data/
│ ├── UNSW_NB15_training-set.csv
│ └── UNSW_NB15_testing-set.csv
├── saved_models/
│ ├── cnn_ids_model.h5
│ ├── ids_preprocessor.pkl
│ ├── label_encoder.pkl
│ └── original_feature_names.pkl
├── .gitignore
└── README.md </code> </pre>
## Setup and Installation

1.  **Clone the Repository:**
    Navigate to your desired directory and clone the project:
    ```bash
    git clone [https://github.com/CatOn60Hz/Real-time-Network-Traffic-Classifier-IDS.git](https://github.com/CatOn60Hz/Real-time-Network-Traffic-Classifier-IDS.git)
    cd Real-time-Network-Traffic-Classifier-IDS
    ```

2.  **Create and Activate Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    ```
    Activate the environment:
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    Once your virtual environment is active, install all necessary Python libraries:
    ```bash
    pip install pandas numpy scikit-learn tensorflow flask requests joblib
    ```

4.  **Prepare Dataset:**
    * Download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`.
    * Place both CSV files inside the `data/` folder in your project's root directory (refer to the [Dataset](#dataset) section for more details).


## Usage

### 1. Train and Save the Model

```sh
python train_and_save_model.py
```

This will:
- Preprocess the data
- Train the 1D CNN
- Save the following files to the `saved_models/` directory:
  - `cnn_ids_model.h5`
  - `ids_preprocessor.pkl`
  - `label_encoder.pkl`
  - `original_feature_names.pkl`

---

### 2. Run the Flask API

```sh
python app.py
```

- This starts a Flask server at: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- Keep this terminal running to accept real-time prediction requests.

---

### 3. Simulate Network Traffic

Open a new terminal window and run:

```sh
python simulate_traffic.py
```

- This will randomly select flows from the test dataset.
- Each selected flow will be sent to the `/predict` endpoint.
- Predictions will be printed in the terminal.

---

## Results and Performance

- **Overall Accuracy:** > 99.9% on the test set.
- High performance across all 10 classes (9 attack types + normal).
- Strong precision, recall, and F1-score metrics.
- Classification report and confusion matrix are given below.
![Image](https://github.com/user-attachments/assets/6465958c-102d-4d9f-8c38-760b3a2c78ff)
![Image](https://github.com/user-attachments/assets/af3fe0b3-fdb1-43bd-aca4-7957a66ab0a1)
---

## License

This project is licensed under the MIT License.


