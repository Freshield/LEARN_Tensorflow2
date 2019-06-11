#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a7_tf1_customized_estimator.py
@Time: 2019-06-11 17:26
@Last_update: 2019-06-11 17:26
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

train_file = "data/titanic/train.csv"
eval_file = "data/titanic/eval.csv"

train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)

print(train_df.head())
print(eval_df.head())

y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')

print(train_df.head())
print(eval_df.head())
print(y_train.head())
print(y_eval.head())

print(train_df.describe())
categorical_columns = ['sex', 'n_siblings_spouses', 'parch', 'class',
                       'deck', 'embark_town', 'alone']
numeric_columns = ['age', 'fare']

feature_columns = []
for categorical_column in categorical_columns:
    vocab = train_df[categorical_column].unique()
    print(categorical_column, vocab)
    feature_columns.append(
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                categorical_column, vocab
            )
        )
    )

for numeric_column in numeric_columns:
    feature_columns.append(
        tf.feature_column.numeric_column(
            numeric_column, dtype=tf.float32
        )
    )

def make_dataset(data_df, label_df, epochs=10, shuffle=True,
                 batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(data_df), label_df))

    if shuffle:
        dataset = dataset.shuffle(10000)

    dataset = dataset.repeat(epochs).batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()

output_dir = 'data/customized_estimator'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def model_fn(features, labels, mode, params):
    input_for_next_layer = tf.feature_column.input_layer(
        features, params['feature_columns'])

    for n_unit in params['hidden_units']:
        input_for_next_layer = tf.layers.dense(input_for_next_layer,
                                               units = n_unit,
                                               activation = tf.nn.relu)

    logits = tf.layers.dense(input_for_next_layer,
                             params['n_classes'],
                             activation=None)
