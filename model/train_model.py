"""Module to train keras on mnist"""
import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
import os

from .data_prep import INPUT_SHAPE, NUM_CLASSES, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST


def train_model():
    if not os.path.isfile('trained_mnist.h5'):
        batch_size = 128
        epochs = 1

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=INPUT_SHAPE))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(X_TRAIN, Y_TRAIN,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(X_TEST, Y_TEST))
        score = model.evaluate(X_TEST, Y_TEST, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        model.save('trained_mnist.h5')
        del model
