# coding:utf-8
import keras
from keras import regularizers
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D
from keras import optimizers
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt # 导入可视化的包
import csv
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import time
import keras.utils as np_utils
from keras.models import load_model
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers.core import Activation, Flatten
from keras import backend as K

class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        model.add(Convolution2D(20, kernel_size=5, padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, kernel_size=5, padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(NB_CLASSES))
        model.add(Activation('softmax'))
        model.summary()
        return model

NB_EPOCH = 50
NB_CLASSES = 10
DROPOUT = 0.3
BATCH_SIZE = 128
INPUT_SHAPE = (1, 28, 28)
current_time = time.time()

K.set_image_dim_ordering('th')

train_path = '/home/zlee/PycharmProjects/kaggle_digit_recognizer/data/train_1.csv'

train_matrix = np.loadtxt(open(train_path), delimiter=",", skiprows=0)

X = train_matrix[:, 0:-1]
Y = train_matrix[:, 0]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.001, random_state=42)

x_train.astype('float32')
x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)


y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)


model.compile(optimizer=Adam(), loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_data=(x_test, y_test))
model.save('./models/1.7_model.h5')

score = model.evaluate(x_test, y_test)

print("loss:",score[0])
print("accu:",score[1])


