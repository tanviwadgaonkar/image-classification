import numpy as np
from data_preprocessing import preprocess_data
from model import build_model

def train_model(data_dir):
    images, labels = preprocess_data(data_dir)
    model = build_model(input_shape=(224, 224, 3))
    model.fit(images, labels, epochs=10, validation_split=0.2)
    model.save('model.h5')

if __name__ == "__main__":
    train_model('data/processed/')
