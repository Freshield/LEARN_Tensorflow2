#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a1_embedding_padding_pooling.py
@Time: 2019-08-01 10:32
@Last_update: 2019-08-01 10:32
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

from tensorflow import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)


imdb = keras.datasets.imdb
vocab_size = 10000
index_from = 3
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    path='imdb.npz', num_words=vocab_size, index_from=index_from)

# train_data 是不定长句子的list, 里边都是数字，整体是包含list的list
# train_data[0] 是一个句子，里边都是数字，整体是包含数字的list
print(train_data[0], train_labels[0])
print(train_data.shape, train_labels.shape)
print(len(train_data[0]), len(train_data[1]))
print(test_data.shape, test_labels.shape)

word_index = imdb.get_word_index()
print(word_index)
print(len(word_index))

# 是word: id的字典
word_index = {k:(v+3) for k, v in word_index.items()}

word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<END>'] = 3

# 是id: word的字典
reverse_word_index = {v: k for k, v in word_index.items()}

def decode_view(text_ids):
    return ' '.join(
        [reverse_word_index.get(word_id, '<UNK>') for word_id in text_ids]
    )

print(decode_view(train_data[0]))

max_length = 500

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index['<PAD>'],
    padding='post',
    maxlen=max_length
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index['<PAD>'],
    padding='post',
    maxlen=max_length
)

print(train_data[0])
print(train_data.shape)

# 一个字符的矩阵长度
embedding_dim = 16
# 一次取几个句子
batch_size = 16

model = keras.models.Sequential([
    # 1. define matrix: [vocab_size, embedding_dim]
    # vocab_size是一共有多少个单词
    # 2. [1,2,3,4..], max_length * embedding_dim
    # 3. batch_size * max_length * embedding_dim
    # input_length就是max_length就是一个句子的长度
    # 输入是 batch, max_length,
    # embedding matrix是 vocab_size, embedding_dim
    # Embedding层会把max_length矩阵中每个词替换成embedding matrix中的相应向量
    # 输出就变成了batch, max_length, embedding_dim
    keras.layers.Embedding(vocab_size, embedding_dim,
                           input_length=max_length),
    # batch_size * max_length * embedding_dim
    #   -> batch_size * embedding_dim
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels,
                    epochs=30,
                    batch_size=batch_size,
                    validation_split=0.2)

def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_'+label] = history.history['val_'+label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()

plot_learning_curves(history, 'acc', 30, 0, 1)
plot_learning_curves(history, 'loss', 30, 0, 1)

print(model.evaluate(
    test_data, train_labels,
    batch_size=batch_size
))