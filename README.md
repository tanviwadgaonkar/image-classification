# Image Classification using Convolutional Neural Networks (CNNs)

A complete machine learning pipeline for classifying images using a custom CNN model.
This project covers data preprocessing, model training, evaluation, and utilities for running experiments.

### Features

End-to-end image classification workflow

Data preprocessing with augmentation

Custom CNN model

Training loop with loss/accuracy tracking

Evaluation script

Jupyter notebooks for analysis

Unit tests for all modules

### Tech Stack

Python

NumPy, Pandas

TensorFlow / Keras or pure Python CNN

Matplotlib / Seaborn

scikit-learn

### 📁 Folder Structure

image-classification/

│

├── notebooks/

│   ├── data_exploration.ipynb

│   ├── model_training.ipynb

│   └── results_analysis.ipynb

│


├── src/

│   ├── data_preprocessing.py     # Dataset loading + augmentation

│   ├── model.py                  # CNN architecture

│   ├── train.py                  # Training pipeline


│   ├── evaluate.py               # Evaluation script

│   └── utils.py                  # Helper functions


│


├── tests/


│   ├── test_data_preprocessing.py


│   ├── test_evaluate.py


│   ├── test_model.py


│   └── test_train.py


│


├── requirements.txt

├── setup.py

└── README.md

## 🛠️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Preprocess the Dataset
```bash
python src/data_preprocessing.py
```

### 3️⃣ Train the Model
```bash
python src/train.py
```

### 4️⃣ Evaluate the Model
```bash
python src/evaluate.py
```

### 5️⃣ Explore Results (Optional)
Use the Jupyter notebooks for:

- Data exploration  
- Model training visualization  
- Result analysis  

```bash
jupyter notebook notebooks/
```
### My Contribution

Implemented preprocessing + augmentation pipeline

Designed and trained the CNN architecture

Developed evaluation scripts and metrics

Created training and analysis notebooks

Wrote unit tests for every module

### 📌 Notes

This project is for learning and experimentation.
Not intended for production deployment.


