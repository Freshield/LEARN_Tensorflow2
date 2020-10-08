# coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: a7_save_and_load.py
@Time: 2020-10-06 16:40
@Last_update: 2020-10-06 16:40
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import os

import tensorflow as tf
from tensorflow import keras


print(tf.version.VERSION)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784, )),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


model = create_model()

model.summary()

checkpoint_path = 'data/training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
#
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path, save_weights_only=True, verbose=1)
#
# model.fit(train_images, train_labels, epochs=10,
#           validation_data=(test_images, test_labels),
#           callbacks=[cp_callback])

# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print(f'Untrained model, accuracy: {100*acc:5.2f}%')
# model.load_weights(checkpoint_path)
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print(f'Restored model, accuracy: {100*acc:5.2f}%')

checkpoint_path = 'data/training_2/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5
)
# model.save_weights(checkpoint_path.format(epoch=0))
# model.fit(train_images, train_labels, epochs=50,
#           callbacks=[cp_callback],
#           validation_data=(test_images, test_labels),
#           verbose=0)

# latest = tf.train.latest_checkpoint(checkpoint_dir)
# print(latest)
# model.load_weights(latest)
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print(f'Restored model, accuracy: {100*acc:5.2f}%')

# print('\nload weights')
# model.save_weights('data/checkpoints/my_checkpoint')
# model = create_model()
# model.load_weights('data/checkpoints/my_checkpoint')
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print(f'Restored model, accuracy: {100*acc:5.2f}%')


# model.fit(train_images, train_labels, epochs=5)
# model.save('data/saved_model/my_model')
# new_model = tf.keras.models.load_model('data/saved_model/my_model')
# new_model.summary()
# loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
# print(f'Restored model, accuracy: {100*acc:5.2f}%')
# print(new_model.predict(test_images).shape)

# model.fit(train_images, train_labels, epochs=5)
# model.save('data/my_model.h5')
new_model = tf.keras.models.load_model('data/my_model.h5')
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print(f'Restored model, accuracy: {100*acc:5.2f}%')
