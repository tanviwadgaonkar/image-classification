"""CNN architecture.

Preprocessing lives in the model: ``Rescaling`` maps [0, 255] -> [0, 1] and
the augmentation layers are active only during ``fit``. A saved model
therefore accepts raw RGB pixels at inference time, exactly as in training.

Global average pooling replaces Flatten: on 224x224 inputs Flatten -> Dense
had ~12M parameters in one layer and overfit small datasets badly.
"""
from __future__ import annotations


def build_model(input_shape: tuple[int, int, int], num_classes: int, learning_rate: float = 1e-3):
    import tensorflow as tf
    from tensorflow.keras import layers

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    augment = tf.keras.Sequential(
        [layers.RandomFlip("horizontal"), layers.RandomRotation(0.05), layers.RandomZoom(0.1)],
        name="augmentation",
    )

    def block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return layers.MaxPooling2D()(x)

    inputs = tf.keras.Input(shape=input_shape)
    x = augment(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128):
        x = block(x, filters)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",  # integer labels from the folder loader
        metrics=["accuracy"],
    )
    return model
