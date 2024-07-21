import os
import cv2
import numpy as np

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize image to 224x224
    image = image / 255.0  # Normalize image
    return image

def preprocess_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            image = load_and_preprocess_image(image_path)
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)
