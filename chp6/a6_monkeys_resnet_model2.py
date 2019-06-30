
import numpy as np
import pandas as pd

import os

data_dir = '/media/freshield/SSD_1T/Data/a4_tensorflow2/monkey_10s'
print(os.listdir(data_dir))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn
import sys
import tensorflow as tf
import time

from tensorflow.python.keras.api._v2 import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)


train_dir = os.path.join(data_dir, 'training')
valid_dir = os.path.join(data_dir, 'validation')
label_file = os.path.join(data_dir, 'monkey_labels.txt')
print(os.path.exists(train_dir))
print(os.path.exists(valid_dir))
print(os.path.exists(label_file))

print(os.listdir(train_dir))
print(os.listdir(valid_dir))

lables = pd.read_csv(label_file, header=0)
print(lables)

height = 224
width = 224
channels = 3
batch_size = 24
num_classes = 10

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.resnet50.preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(height, width),
    batch_size=batch_size, seed=7, shuffle=True, class_mode='categorical'
)

valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.resnet50.preprocess_input
)
valid_generator = valid_datagen.flow_from_directory(
    valid_dir, target_size=(height, width),
    batch_size=batch_size, seed=7, shuffle=False, class_mode='categorical'
)

train_num = train_generator.samples
valid_num = valid_generator.samples
print(train_num, valid_num)

# for i in range(2):
#     x, y = train_generator.next()
#     print(y)
#     print(x.shape, y.shape)

resnet50 = keras.applications.ResNet50(include_top=False,
                                       pooling='avg',
                                       weights='imagenet')

for layer in resnet50.layers[0:-5]:
    layer.trainable = False

resnet50_new = keras.models.Sequential([
    resnet50,
    keras.layers.Dense(num_classes, activation='softmax')
])

resnet50_new.compile(loss='categorical_crossentropy',
                     optimizer='sgd', metrics=['accuracy'])
resnet50_new.summary()

epochs = 10
history = resnet50_new.fit_generator(
    train_generator, steps_per_epoch=train_num//batch_size,
    epochs=epochs, validation_data=valid_generator,
    validation_steps=valid_num//batch_size
)

def plot_learning_rate(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_'+label] = history.history['val_'+label]
    pd.DataFrame(data).plot(figsize=(8,5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()

plot_learning_rate(history, 'accuracy', epochs, 0, 1)
plot_learning_rate(history, 'loss', epochs, 0, 2)

