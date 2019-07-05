#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a1_tf_basic_api.py
@Time: 2019-07-05 10:49
@Last_update: 2019-07-05 10:49
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

t = tf.constant([[1,2,3], [4,5,6]])

print(t)
print(t[:,1:])
print(t[...,1])

print(t+10)
print(tf.square(t))
print(t @ tf.transpose(t))

print()
print(t.numpy())
print(np.square(t))
np_t = np.array([[1,2,3], [4,5,6]])
print(tf.constant(np_t))

print()
t = tf.constant(2.718)
print(t.numpy())
print(t.shape)

print()
t = tf.constant('cafe')
print(t)
print(tf.strings.length(t))
print(tf.strings.length(t, unit='UTF8_CHAR'))
print(tf.strings.unicode_decode(t, 'UTF8'))

print()
t = tf.constant(['cafe', 'coffee', '咖啡'])
print(tf.strings.length(t, unit='UTF8_CHAR'))
r = tf.strings.unicode_decode(t, 'UTF8')
print(r)

print()
r = tf.ragged.constant([[11,12], [21,22,23], [], [41]])
print(r)
print(r[1])
print(r[1:2])

print()
r2 = tf.ragged.constant([[51,52], [], [71]])
print(tf.concat([r, r2], axis=0))

print()
r3 = tf.ragged.constant([[13,14], [15], [], [42,43]])
print(tf.concat([r, r3], axis=1))

print()
print(r.to_tensor())

print()
s = tf.SparseTensor(indices=[[0,1], [1,0], [2,3]],
                    values=[1,2,3],
                    dense_shape=[3,4])
print(s)
print(tf.sparse.to_dense(s))

print()
s2 = s * 2
print(s2)

s4 = tf.constant([[10,20],
                  [30,40],
                  [50,60],
                  [70,80]])
print(tf.sparse.sparse_dense_matmul(s, s4))

print()
s5 = tf.SparseTensor(indices=[[0,2], [0,1], [2,3]],
                     values=[1,2,3],
                     dense_shape=[3,4])
print(s5)
s6 = tf.sparse.reorder(s5)
print(tf.sparse.to_dense(s6))

print()
v = tf.Variable([[1,2,3], [4,5,6]])
print(v)
print(v.value())
print(v.numpy())

print()
v.assign(2*v)
print(v.numpy())
v[0,1].assign(42)
print(v.numpy())
v[1].assign([7,8,9])
print(v.numpy())

print()
