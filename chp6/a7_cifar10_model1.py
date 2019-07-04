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

class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

data_dir = '/media/freshield/SSD_1T/Data/a4_tensorflow2/cifar10/'
train_labels_file = os.path.join(data_dir, 'trainLabels.csv')
test_csv_file = os.path.join(data_dir, 'sampleSubmission.csv')
train_folder = os.path.join(data_dir, 'train')
test_folder = os.path.join(data_dir, 'test')

def parse_csv_file(filepath, folder):
    results = []
    with open(filepath, 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        print(line)
        image_id, label_str = line.strip('\n').split(',')
        image_full_path = os.path.join(folder, image_id + '.png')
        results.append((image_full_path, label_str))

    return results

train_labels_info = parse_csv_file(train_labels_file, train_folder)

print(train_labels_info[0:5])
print(len(train_labels_info))

train_df = pd.DataFrame(train_labels_info[:45000])
valid_df = pd.DataFrame(train_labels_info[45000:])

train_df.columns = ['filepath', 'class']
valid_df.columns = ['filepath', 'class']

print(train_df.head())
print(valid_df.head())

height = 32
width = 32
channels = 3
batch_size = 32
num_classes = 10

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=data_dir,
    x_col='filepath',
    y_col='class',
    classes=class_names,
    target_size=(height, width),
    batch_size=batch_size,
    seed=7,
    shuffle=True,
    class_mode='sparse'
)

valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
valid_generator = valid_datagen.flow_from_dataframe(
    valid_df,
    directory=data_dir,
    x_col='filepath',
    y_col='class',
    classes=class_names,
    target_size=(height, width),
    batch_size=batch_size,
    seed=7,
    shuffle=False,
    class_mode='sparse'
)

train_num = train_generator.samples
valid_num = valid_generator.samples

print(train_num, valid_num)

for i in range(2):
    x, y = train_generator.next()
    print(y)
    print(x.shape, y.shape)

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=3, padding='same',
                        activation='relu',
                        input_shape=[width, height, channels]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=128, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=2),

    keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

epochs = 20
history = model.fit_generator(train_generator,
                              steps_per_epoch=train_num//batch_size,
                              epochs=epochs,
                              validation_data=valid_generator,
                              validation_steps=valid_num//batch_size)


def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()


plot_learning_curves(history, 'accuracy', epochs, 0, 1)
plot_learning_curves(history, 'loss', epochs, 0, 2)