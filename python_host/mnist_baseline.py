from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
import pickle
import numpy as np


def main():
    num_classes = 10

    f = open('mnist.pkl', 'rb')
    data = pickle.load(f, encoding='bytes')
    f.close()

    (x_train, y_train), (x_test, y_test) = data
    img_rows, img_cols = 28, 28
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # model = Sequential()
    # model.add(Flatten(input_shape=(28, 28)))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(10, activation='sigmoid'))

    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(16, kernel_size=(5, 5),activation="relu"),
            MaxPooling2D(pool_size=(4, 4)),
            Flatten(),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = 128
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    model.save('./mnist_model/mnist_lr.h5')
    model.save('./mnist_model')


if __name__ == '__main__':
    main()
