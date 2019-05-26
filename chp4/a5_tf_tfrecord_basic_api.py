#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a5_tf_tfrecord_basic_api.py
@Time: 2019-05-25 14:17
@Last_update: 2019-05-25 14:17
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
favorite_books = [name.encode('utf-8') for name in ['machine_learning', 'cc150']]
favorite_books_bytelist = tf.train.BytesList(value=favorite_books)
print(favorite_books_bytelist)

print()
hours_floatlist = tf.train.FloatList(value=[15.5,9.5,7.0,8])
print(hours_floatlist)

print()
age_int64list = tf.train.Int64List(value=[42])
print(age_int64list)

print()
features = tf.train.Features(
    feature = {
        'favorite_books': tf.train.Feature(bytes_list=favorite_books_bytelist),
        'hours': tf.train.Feature(float_list=hours_floatlist),
        'age': tf.train.Feature(int64_list=age_int64list)
    }
)
print(features)

print()
example = tf.train.Example(features=features)
print(example)

serialized_example = example.SerializeToString()
print(serialized_example)

print()
output_dir = 'data/tfrecord_basic'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
filename = 'test.tfrecords'
filename_fullpath = os.path.join(output_dir, filename)
with tf.io.TFRecordWriter(filename_fullpath) as writer:
    for i in range(3):
        writer.write(serialized_example)

print()
dataset = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset:
    print(serialized_example_tensor)

expected_features = {
    'favorite_books': tf.io.VarLenFeature(dtype=tf.string),
    'hours': tf.io.VarLenFeature(dtype=tf.float32),
    'age': tf.io.FixedLenFeature([], dtype=tf.int64)
}
dataset = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset:
    example = tf.io.parse_single_example(
        serialized_example_tensor, expected_features
    )
    books = tf.sparse.to_dense(example['favorite_books'], default_value=b'')

    for book in books:
        print(book.numpy().decode('utf8'))

print()
filename_fullpath_zip = filename_fullpath + '.zip'
options = tf.io.TFRecordOptions(compression_type='GZIP')
with tf.io.TFRecordWriter(filename_fullpath_zip, options=options) as writer:
    for i in range(3):
        writer.write(serialized_example)

dataset_zip = tf.data.TFRecordDataset([filename_fullpath_zip], compression_type='GZIP')

for serialized_example_tensor in dataset:
    example = tf.io.parse_single_example(
        serialized_example_tensor, expected_features
    )
    books = tf.sparse.to_dense(example['favorite_books'], default_value=b'')

    for book in books:
        print(book.numpy().decode('utf8'))