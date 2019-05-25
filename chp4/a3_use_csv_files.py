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


def parse_csv_line(line, n_fields=9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])

    return x, y

print(parse_csv_line(b'-0.9868720801669367,0.832863080552588,-0.18684708416901633,-0.14888949288707784,-0.4532302419670616,-0.11504995754593579,1.6730974284189664,-0.7465496877362412,1.138',
               n_fields=9))
print()

def csv_reader_dataset(
        filenames, n_readers=5, batch_size=32, n_parse_threads=5, shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length=n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)

    return dataset

train_set = csv_reader_dataset(train_filenames, batch_size=3)
for x_batch, y_batch in train_set.take(2):
    print('x:', x_batch)
    print('y:', y_batch)
print()

batch_size = 32
train_set = csv_reader_dataset(train_filenames, batch_size=batch_size)
valid_set = csv_reader_dataset(valid_filenames, batch_size=batch_size)
test_set = csv_reader_dataset(test_filenames, batch_size=batch_size)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=[8]),
    keras.layers.Dense(1)
])
model.compile(loss='mse', optimizer='sgd')
callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2
)]

history = model.fit(train_set, validation_data=valid_set,
                    steps_per_epoch=11160 // batch_size,
                    validation_steps=3870 // batch_size,
                    epochs=100,
                    callbacks=callbacks)

model.evaluate(test_set, steps=5160//batch_size)

