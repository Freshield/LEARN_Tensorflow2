#coding=utf-8
"""
@Author: Freshield
@License: (C) Copyright 2018, BEIJING LINKING MEDICAL TECHNOLOGY CO., LTD.
@Contact: yangyufresh@163.com
@File: a4_tf_function_and_auto_graph.py
@Time: 2019-05-07 16:26
@Last_update: 2019-05-07 16:26
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

def scaled_elu(z, scale=1.0, alpha=1.0):
    is_positive = tf.greater_equal(z, 0.0)
    return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))

print(scaled_elu(tf.constant(-3.)))
print(scaled_elu(tf.constant([-3., -2.5])))

scaled_elu_tf = tf.function(scaled_elu)
print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3., -2.5])))

print(scaled_elu_tf.python_function is scaled_elu)

# begin_time = time.time()
# for i in range(100):
#     scaled_elu(tf.random.normal((1000, 1000)))
# print(time.time() - begin_time)
#
# begin_time = time.time()
# for i in range(100):
#     scaled_elu_tf(tf.random.normal((1000, 1000)))
# print(time.time() - begin_time)

@tf.function
def converge_to_2(n_iters):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iters):
        total += increment
        increment /= 2.0
    return total

print(converge_to_2(20))

def display_tf_code(func):
    code = tf.autograph.to_code(func)
    print(code)

display_tf_code(scaled_elu)

var = tf.Variable(0.)

@tf.function
def add_21(input):
    return input.assign_add(21)

print(add_21(var))

@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name='x')])
def cube(z):
    return tf.pow(z, 3)

print(cube(tf.constant([1,2,3])))
print(cube)

cube_func_int32 = cube.get_concrete_function(
    tf.TensorSpec([None], tf.int32)
)
print(cube_func_int32)

print(cube_func_int32 is cube.get_concrete_function(
    tf.TensorSpec([5], tf.int32)))
print(cube_func_int32 is cube.get_concrete_function(
    tf.constant([1, 2, 3])))

print(cube_func_int32.graph)

print(cube_func_int32.graph.get_operations())

pow_op = cube_func_int32.graph.get_operations()[2]
print(pow_op)

print(pow_op.inputs)
print(pow_op.outputs)

print(cube_func_int32.graph.get_operation_by_name('x'))
print(cube_func_int32.graph.get_tensor_by_name('x:0'))

print(cube_func_int32.graph.as_graph_def())