#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a1_tf_data_basic_api.py
@Time: 2019-07-10 10:19
@Last_update: 2019-07-10 10:19
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

from tensorflow.python.keras.api._v2 import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)


print()
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
print(dataset)

for item in dataset:
    print(item)


print()
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)


print()
dataset2 = dataset.interleave(
    lambda v: tf.data.Dataset.from_tensor_slices(v),
    cycle_length=5,
    block_length=5
)
for item in dataset2:
    print(item)

print()
x = np.array([[1,2],[3,4],[5,6]])
y = np.array(['cat','dog','fox'])
dataset3 = tf.data.Dataset.from_tensor_slices((x,y))
print(dataset3)

for item_x, item_y in dataset3:
    print(item_x.numpy(), item_y.numpy())

print()
dataset4 = tf.data.Dataset.from_tensor_slices({
    'feature': x, 'label': y
})
for item in dataset4:
    print(item['feature'].numpy(), item['label'].numpy())