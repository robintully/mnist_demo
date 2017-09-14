"""Preps data, loads it in and formats it"""
import keras
from keras.datasets import mnist

# input image dimensions
IMG_ROWS, IMG_COLS = 28, 28
NUM_CLASSES = 10

(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = mnist.load_data()


X_TRAIN = X_TRAIN.reshape(X_TRAIN.shape[0], IMG_ROWS, IMG_COLS, 1)
X_TEST = X_TEST.reshape(X_TEST.shape[0], IMG_ROWS, IMG_COLS, 1)
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)

X_TRAIN = X_TRAIN.astype('float32')
X_TEST = X_TEST.astype('float32')
X_TRAIN /= 255
X_TEST /= 255
Y_TRAIN = keras.utils.to_categorical(Y_TRAIN, NUM_CLASSES)
Y_TEST = keras.utils.to_categorical(Y_TEST, NUM_CLASSES)
