#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a6_tf_data_generate_tfrecord.py
@Time: 2019-05-26 14:43
@Last_update: 2019-05-26 14:43
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
source_dir = 'data/generate_csv/'

def get_filenames_by_prefix(source_dir, prefix_name):
    all_files = os.listdir(source_dir)
    results = []
    for filename in all_files:
        if filename.startswith(prefix_name):
            results.append(os.path.join(source_dir, filename))

    return results


train_filenames = get_filenames_by_prefix(source_dir, 'train')
valid_filenames = get_filenames_by_prefix(source_dir, 'valid')
test_filenames = get_filenames_by_prefix(source_dir, 'test')

print(train_filenames)
print(valid_filenames)
print(test_filenames)

print()
def parse_csv_line(line, n_fields=9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])

    return x, y

def csv_reader_dataset(
        filenames, n_readers=5, batch_size=32, n_parse_threads=5, shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length=5
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)

    return dataset

batch_size = 32
train_set = csv_reader_dataset(train_filenames, batch_size=batch_size)
valid_set = csv_reader_dataset(valid_filenames, batch_size=batch_size)
test_set = csv_reader_dataset(test_filenames, batch_size=batch_size)

def serialize_example(x, y):
    input_features = tf.train.FloatList(value=x)
    label = tf.train.FloatList(value=y)
    features = tf.train.Features(
        feature = {
            'input_features': tf.train.Feature(float_list=input_features),
            'label': tf.train.Feature(float_list=label)
        }
    )
    example = tf.train.Example(features = features)
    
    return example.SerializeToString()

def csv_dataset_to_tfrecords(
        base_filename, dataset, n_shards, steps_per_shard, compression_type=None):
    options = tf.io.TFRecordOptions(compression_type=compression_type)
    all_filenames = []
    for shard_id in range(n_shards):
        filename_fullpath = '{}_{:05d}_of_{:05d}'.format(base_filename, shard_id, n_shards)
        with tf.io.TFRecordWriter(filename_fullpath, options) as writer:
            for x_batch, y_batch in dataset.take(steps_per_shard):
                for x_example, y_example in zip(x_batch, y_batch):
                    writer.write(serialize_example(x_example, y_example))
                    
        all_filenames.append(filename_fullpath)
        
    return all_filenames

n_shards = 20
train_steps_per_shard = 11610 // batch_size // n_shards
valid_steps_per_shard = 3880 // batch_size // n_shards
test_steps_per_shard = 5170 // batch_size // n_shards

output_dir = 'data/generate_tfrecords_zip'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
train_basename = os.path.join(output_dir, 'train')
valid_basename = os.path.join(output_dir, 'valid')
test_basename = os.path.join(output_dir, 'test')

# train_tfrecord_filenames = csv_dataset_to_tfrecords(
#     train_basename, train_set, n_shards, train_steps_per_shard, compression_type='GZIP')
# valid_tfrecord_filenames = csv_dataset_to_tfrecords(
#     valid_basename, valid_set, n_shards, valid_steps_per_shard, compression_type='GZIP')
# test_tfrecord_fielnames = csv_dataset_to_tfrecords(
#     test_basename, test_set, n_shards, test_steps_per_shard, compression_type='GZIP')
#
# print(train_tfrecord_filenames)
# print(valid_tfrecord_filenames)
# print(test_tfrecord_fielnames)
# print()
train_tfrecord_filenames = [os.path.join(output_dir, filename) for filename in os.listdir(output_dir) if 'train' in filename]
valid_tfrecord_filenames = [os.path.join(output_dir, filename) for filename in os.listdir(output_dir) if 'valid' in filename]
test_tfrecord_filenames = [os.path.join(output_dir, filename) for filename in os.listdir(output_dir) if 'test' in filename]

expected_features = {
    'input_features': tf.io.FixedLenFeature([8], dtype=tf.float32),
    'label': tf.io.FixedLenFeature([1], dtype=tf.float32)
}

def parse_example(serialized_example):
    example = tf.io.parse_single_example(
        serialized_example, expected_features
    )
    return example['input_features'], example['label']

def tfrecords_reader_dataset(
        filenames, n_readers=5, batch_size=32, n_parse_threads=5, shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(
            filename, compression_type='GZIP'),
        cycle_length=n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_example, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)

    return dataset

# print()
# tfrecords_train = tfrecords_reader_dataset(train_tfrecord_filenames, batch_size=3)
# for x_batch, y_batch in tfrecords_train.take(2):
#     print(x_batch)
#     print(y_batch)

print()
batch_size = 32
tfrecords_train_set = tfrecords_reader_dataset(train_tfrecord_filenames, batch_size=batch_size)
tfrecords_valid_set = tfrecords_reader_dataset(valid_tfrecord_filenames, batch_size=batch_size)
tfrecords_test_set = tfrecords_reader_dataset(test_tfrecord_filenames, batch_size=batch_size)


model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=[8]),
    keras.layers.Dense(1)
])
model.compile(loss='mse', optimizer='sgd')
callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2
)]

history = model.fit(
    tfrecords_train_set, validation_data=tfrecords_valid_set,
    steps_per_epoch=11160//batch_size, validation_steps=3870//batch_size,
    epochs=100, callbacks=callbacks)

model.evaluate(tfrecords_test_set, steps=5160//batch_size)