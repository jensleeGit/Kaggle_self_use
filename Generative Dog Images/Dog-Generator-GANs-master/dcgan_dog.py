# coding:utf-8
import os
#import keras
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers.core import Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
from PIL import Image
import argparse
import math
import numpy as np
from keras import backend as K
import tensorflow as tf
import glob
import cv2
import zipfile

path = './data'
g_epoch_num = 100000


def load_data(path):
    X_train = []
    img_list = glob.glob(path + '/*.jpg')
    for img in img_list:
        _img = cv2.imread(img)
        _img = cv2.resize(_img, (64, 64))
        X_train.append(_img)
    return np.array(X_train, dtype=np.uint8)


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=1000, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*8*8))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((8, 8, 128), input_shape=(8*8*128,)))
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    # model.summary()
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(64, 64, 3)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1], 3), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i * shape[0]: (i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, :]
    return image


def train(BATCH_SIZE):
    # 载入训练参数
    X_train = load_data(path)
    X_train = (X_train.astype(np.float32) - 127.5)/127.5

    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(g_epoch_num):
        print("Epoch is", epoch)

        # 训练一个batchsize里面的数据
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):

            # 产生随机噪声
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 1000))

            # 这里面都是真图片
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

            # 这里产生假图片
            generated_images = g.predict(noise, verbose=0)

            cv2.imshow('generator_dog_image', generated_images[0, :, :, :])
            cv2.waitKey(10)
            # 将真图片与假图片拼接在一起
            X = np.concatenate((image_batch, generated_images))

            # 前64张图片标签为1,即真图，后64张照片为假图
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE

            # 对于判别器进行训练，不断提高判别器的识别精度
            d_loss = d.train_on_batch(X, y)

            # 再次产生随机噪声
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 1000))

            # 设置判别器的参数不可调整
            d.trainable = False

            # 在此我们送入噪声，并认为这些噪声是真实的标签
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)

            # 此时设置判别器可以被训练，参数可以被修改
            d.trainable = True

            # 打印损失值
            print("Epoch is : %s, batch %d d_loss : %s, g_loss : %f" % (epoch, index, d_loss, g_loss))
            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    # print(nice)
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 1000))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        for i in range(16):
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 1000))
            generated_images = g.predict(noise, verbose=1)
            img = np.zeros((generated_images.shape[0], generated_images.shape[1], 3), dtype=np.uint8)
            for j in range(0, 64):
                img = generated_images[j, :, :, :]
                cv2.imshow('test1', img)
                cv2.waitKey(10)
                cv2.imwrite("./dog_images/generated_image" + str(i) + '_' + str(j) + ".png", img*127.5+127.5)

            print('i : i', i)

if __name__ == "__main__":
    train(BATCH_SIZE=64)
    generate(BATCH_SIZE=64, nice=False)
