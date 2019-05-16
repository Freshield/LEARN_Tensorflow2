#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a3_use_csv_files.py
@Time: 2019-07-10 11:04
@Last_update: 2019-07-10 11:04
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

data_dir = 'data/generate_csv'
train_filenames = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if 'train' in filename]
train_filenames.sort()
valid_filenames = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if 'valid' in filename]
valid_filenames.sort()
test_filenames = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if 'test' in filename]
test_filenames.sort()

filename_dataset = tf.data.Dataset.list_files(train_filenames)

n_readers = 5
dataset = filename_dataset.interleave(
    lambda filename: tf.data.TextLineDataset(filename).skip(1),
    cycle_length=5
)
for line in dataset.take(5):
    print(line.numpy())