#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a5_text_generation_lstm.py
@Time: 2019-07-11 14:51
@Last_update: 2019-07-11 14:51
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

input_filepath = 'data/shakespeare.txt'
text = open(input_filepath, 'r').read()

print(len(text))
print(text[0:100])

vocab = sorted(set(text))
print(len(vocab))
print(vocab)

char2idx = {char: idx for idx, char in enumerate(vocab)}
print(char2idx)

idx2char = np.array(vocab)
print(idx2char)

text_as_int = np.array([char2idx[c] for c in text])
print(text_as_int[0:10])
print(text[0:10])

def split_input_target(id_text):
    return id_text[0:-1], id_text[1:]

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
seq_length = 100
seq_dataset = char_dataset.batch(seq_length+1, drop_remainder=True)

for ch_id in char_dataset.take(2):
    print(ch_id, idx2char[ch_id.numpy()])

print()
for seq_id in seq_dataset.take(2):
    print(seq_id)
    print(''.join(idx2char[seq_id.numpy()]))

seq_dataset = seq_dataset.map(split_input_target)

for item_input, item_outupt in seq_dataset.take(2):
    print(item_input.numpy())
    print(item_outupt.numpy())

batch_size = 64
buffer_size = 10000

seq_dataset = seq_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

vocab_size = len(vocab)
embedding_dim = 32
rnn_units = 64

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        keras.layers.LSTM(units=rnn_units, stateful=True, recurrent_initializer='glorot_uniform',
                               return_sequences=True),
        keras.layers.Dense(vocab_size)
    ])

    return model

model = build_model(
    vocab_size, embedding_dim, rnn_units, batch_size)

model.summary()

for input_example_batch, target_example_batch in seq_dataset.take(1):
    example_batch_preditions = model(input_example_batch)
    print(example_batch_preditions.shape)

sample_indices = tf.random.categorical(
    logits=example_batch_preditions[0], num_samples=1
)
print(sample_indices)
sample_indices = tf.squeeze(sample_indices, axis=-1)
print(sample_indices)

print('Input:', repr(''.join(idx2char[input_example_batch[0]])))
print()
print('Output:', repr(''.join(idx2char[target_example_batch[0]])))
print()
print('Predictions:', repr(''.join(idx2char[sample_indices])))

def loss(labels, logits):
    return keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

example_loss = loss(target_example_batch, example_batch_preditions)
print(example_loss.shape)
print(example_loss.numpy().mean())

output_dir = 'data/text_generation_lstm3_checkpoints'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
checkpoint_prefix = os.path.join(output_dir, 'ckpt_{epoch}')
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

epochs = 5
history = model.fit(seq_dataset, epochs=epochs, callbacks=[checkpoint_callback])

print(tf.train.latest_checkpoint)

model2 = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model2.load_weights(tf.train.latest_checkpoint(output_dir))
model2.build(tf.TensorShape([1, None]))
model2.summary()

def generate_text(model, start_string, num_generate = 1000):
    input_eval = [char2idx[ch] for ch in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    model.reset_states()

    tempreature = 2

    for _ in range(num_generate):

        predictions = model(input_eval)

        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / tempreature

        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        text_generated.append(idx2char[predicted_id])

        input_eval = tf.expand_dims([predicted_id], 0)

    return start_string + ''.join(text_generated)

new_text = generate_text(model2, 'All: ')
print(new_text)
