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
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print(f'Version: {tf.__version__}')
print(f'Eager mode: {tf.executing_eagerly()}')
print(f'Hub version: {hub.__version__}')
print(tf.config.experimental.list_physical_devices("GPU"))
print(f'GPUS is {"avaliable" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVALIABLE"}')