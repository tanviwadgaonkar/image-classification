import numpy as np
import tensorflow as tf
from data_preprocessing import preprocess_data

def evaluate_model(data_dir):
    images, labels = preprocess_data(data_dir)
    model = tf.keras.models.load_model('model.h5')
    loss, accuracy = model.evaluate(images, labels)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    evaluate_model('data/processed/')
