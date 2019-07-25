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
# import tensorflow as tf
import cv2

# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#gpu_options = tf.GPUOptions(allow_growth=True)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# 生成网络
def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(7*7*128,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    # model.summary()
    return model

# 分辨网络
def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1))
            )
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

# 生成包含分辨网络
def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    # 此时不改变判别器参数，只改变生成器参数
    d.trainable = False
    model.add(d)
    model.summary()
    return model

# 组合图像
def combine_images(generated_images):

    # num为batch_size
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))

    # shape为图像的大小
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i * shape[0]: (i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, 0]
    return image


def train(BATCH_SIZE):

    # 取出mnist的训练数据与测试数据
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # 归一化
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]

    # 声明分辨网络模型
    d = discriminator_model()

    # 声明产生模型
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)

    # 二者的优化函数
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    # 模型编译
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    # 训练100轮
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):

            # 产生随机噪声
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

            # 一个batch的图片
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

            # 生成器产生的图片，不进行显示
            generated_images = g.predict(noise, verbose=0)
            # print('generated_images.shape : ', generated_images.shape)

            cv2.imshow('test', generated_images[1, :, :, :])
            cv2.waitKey(1)

            # 如果一个epoch中，循环了20次
            # 讲多张图片组成一个拼图，并进行存储
            if index % 20 == 0:
                image = combine_images(generated_images)
                cv2.imshow('test2', image)
                cv2.waitKey(1)
                image = image*127.5+127.5
                # 预测的图像进行存储
                Image.fromarray(image.astype(np.uint8)).save('mnist_images/' + str(epoch)+"_"+str(index)+".png")

            # 将真实图片与模拟图片进行拼接,(128, 28, 28, 1)
            X = np.concatenate((image_batch, generated_images))

            # y是标签，前64个为真，后64个为假
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE

            # 送入分辨器进行训练
            d_loss = d.train_on_batch(X, y)

            print("batch %d d_loss : %f" % (index, d_loss))

            # 随机生成初始噪音图片
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False

            # 生成器生成的图片，一直以为是真的
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)

            d.trainable = True
            print("batch %d    g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
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
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # 获取参数
    args = get_args()

    # 开始训练
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
