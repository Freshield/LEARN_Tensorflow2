#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a5_tf1_init_dataset.py
@Time: 2019-07-17 16:23
@Last_update: 2019-07-17 16:23
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow.python import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# x_train: [None, 28, 28] -> [None, 784]
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28 * 28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28 * 28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28 * 28)

y_train = np.asarray(y_train, dtype = np.int64)
y_valid = np.asarray(y_valid, dtype = np.int64)
y_test = np.asarray(y_test, dtype = np.int64)

def make_dataset(images, labels, epochs, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)

    return dataset

batch_size = 20
epochs = 10

images_placeholder = tf.placeholder(tf.float32, [None, 28 * 28])
labels_placeholder = tf.placeholder(tf.int64, [None])

dataset = make_dataset(images_placeholder, labels_placeholder,
                       epochs=epochs, batch_size=batch_size)

dataset_iter = dataset.make_initializable_iterator()
x, y = dataset_iter.get_next()

with tf.Session() as sess:
    sess.run(dataset_iter.initializer,
             feed_dict={
                 images_placeholder: x_train_scaled,
                 labels_placeholder: y_train
             })
    x_val, y_val = sess.run([x, y])
    print(x_val.shape)
    print(y_val.shape)

    sess.run(dataset_iter.initializer,
             feed_dict={
                 images_placeholder: x_valid_scaled,
                 labels_placeholder: y_valid
             })
    x_val, y_val = sess.run([x, y])
    print(x_val.shape)
    print(y_val.shape)
