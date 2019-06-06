#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a2_tf_premade_estimators.py
@Time: 2019-06-06 14:37
@Last_update: 2019-06-06 14:37
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


train_file = 'data/titanic/train.csv'
eval_file = 'data/titanic/eval.csv'

train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)
print()
y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')

print(train_df.head())
print(eval_df.head())
print(y_train.head())
print(y_eval.head())

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

def make_dataset(data_df, label_df, epochs=10, shuffle=True, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(data_df), label_df))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)

    return dataset

output_dir = 'data/baseline_model'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

baseline_estimator = tf.estimator.BaselineClassifier(
    model_dir=output_dir, n_classes=2
)
baseline_estimator.train(input_fn = lambda : make_dataset(train_df, y_train, epochs=100))
result = baseline_estimator.evaluate(input_fn = lambda : make_dataset(eval_df, y_eval, epochs=1, shuffle=False, batch_size=20))
print(result)

# linear_output_dir = 'data/linear_model'
# if not os.path.exists(linear_output_dir):
#     os.mkdir(linear_output_dir)
#
# linear_estimator = tf.estimator.LinearClassifier(
#     model_dir=linear_output_dir, n_classes=2, feature_columns=feature_columns)
# linear_estimator.train(input_fn = lambda : make_dataset(
#     train_df, y_train, epochs=100))
# print(linear_estimator.evaluate(input_fn = lambda : make_dataset(
#     eval_df, y_eval, epochs=1, shuffle=False
# )))

dnn_output_dir = 'data/dnn_model'
if not os.path.exists(dnn_output_dir):
    os.mkdir(dnn_output_dir)

dnn_estimator = tf.estimator.DNNClassifier(
    model_dir=dnn_output_dir, n_classes=2, feature_columns=feature_columns,
    hidden_units=[128, 128], activation_fn=tf.nn.relu, optimizer='Adam')

dnn_estimator.train(input_fn = lambda : make_dataset(
    train_df, y_train, epochs=100))

print(dnn_estimator.evaluate(input_fn = lambda : make_dataset(
    eval_df, y_eval, epochs=1, shuffle=False)))