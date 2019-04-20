#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a10_keras_wide_deep_subclass.py
@Time: 2019-04-20 15:48
@Last_update: 2019-04-20 15:48
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

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

from sklearn.model_selection import train_test_split
x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state=7)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=11)

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

class WideDeepModel(keras.models.Model):
    def __init__(self):
        super(WideDeepModel, self).__init__()

        self.hidden1_layer = keras.layers.Dense(30, activation='relu')
        self.hidden2_layer = keras.layers.Dense(30, activation='relu')
        self.output_layer = keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        hidden1 = self.hidden1_layer(inputs)
        hidden2 = self.hidden2_layer(hidden1)
        concat = keras.layers.concatenate([inputs, hidden2])
        output = self.output_layer(concat)
        return output

model = WideDeepModel()
print(x_train_scaled.shape[-1])
model.build(input_shape=(None, x_train_scaled.shape[-1]))

print(model.summary())
model.compile(loss='mse', optimizer='sgd')
callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2
)]

history = model.fit(x_train_scaled, y_train,
                    validation_data=(x_valid_scaled, y_valid),
                    epochs=100,
                    callbacks=callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)

model.evaluate(x_test_scaled, y_test)