import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


class Model:
    def __init__(self):
        self.imageHeight = 250
        self.imageWidth = 250
        self.classes = 3
        self.model = self.create_model()


    def create_model(self):
        model = Sequential()
        model.add(Input((self.imageHeight, self.imageWidth, 3)))
        model.add(layers.RandomFlip())
        model.add(layers.RandomRotation(0.1))
        model.add(layers.RandomZoom(0.1))
        model.add(Conv2D(16, 3, activation='relu'))
        model.add(MaxPool2D())
        model.add(Conv2D(32, 3, activation='relu'))
        model.add(MaxPool2D())
        model.add(Conv2D(64, 3, activation='relu'))
        model.add(MaxPool2D())
        layers.Dropout(0.2),
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.classes))

        model.compile(optimizer="adam", loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        return model


