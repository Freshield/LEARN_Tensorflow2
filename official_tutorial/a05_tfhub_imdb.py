# coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: a5_tfhub_imdb.py
@Time: 2020-09-27 15:54
@Last_update: 2020-09-27 15:54
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import numpy as np
from tensorflow import keras
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print(f'Version: {tf.__version__}')
print(f'Eager mode: {tf.executing_eagerly()}')
print(f'Hub version: {hub.__version__}')
print(tf.config.experimental.list_physical_devices("GPU"))
print(f'GPUS is {"avaliable" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVALIABLE"}')

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(f'Training entries: {len(train_data)}, labels: {len(train_labels)}')

print(train_data[0])
print(len(train_data[0]), len(train_data[1]))

word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_data[0]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index['<PAD>'],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print(len(train_data[0]), len(test_data[1]))
print(train_data[0])

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
# hub_layer(train_data[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()