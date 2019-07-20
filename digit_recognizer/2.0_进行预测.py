# coding:utf-8
import keras
import os
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.optimizers import SGD # 导入优化函数
import matplotlib.pyplot as plt # 导入可视化的包
import csv
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import time
import keras.utils as np_utils
from keras.models import load_model
import csv

NB_CLASSES = 10

test_path = '/home/zlee/PycharmProjects/kaggle_digit_recognizer/data/test_1.csv'
csv_path = '/home/zlee/PycharmProjects/kaggle_digit_recognizer/data/1.0_submit.csv'
model_path = "./models/1.8_model.h5"
test_matrix = np.loadtxt(open(test_path), delimiter=",", skiprows=0)

x_test = test_matrix


x_test.astype('float32')
x_test = x_test / 255
x_test = x_test.reshape(-1, 1, 28, 28)

model = load_model(model_path)
result = model.predict(x_test)
input_data = np.argmax(result, axis=1)

file = open(csv_path, 'a')

headers = ['ImageId', 'Label']
f_csv = csv.DictWriter(file, headers)

i = 1


for row in input_data:
    rows = [{'ImageId': str(i), 'Label': str(row)}]
    # _str = str(i) + ',' + str(row) + '\n'
    f_csv.writerows(rows)
    i += 1

file.close()
