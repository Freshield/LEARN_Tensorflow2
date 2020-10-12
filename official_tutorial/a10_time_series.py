# coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: a10_time_series.py
@Time: 2020-10-08 09:50
@Last_update: 2020-10-08 09:50
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
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
print(csv_path)

df = pd.read_csv(csv_path)
df = df[5::6]

date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
print(df.head())
print(date_time.head())

# plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
# plot_features = df[plot_cols]
# plot_features.index = date_time
# _ = plot_features.plot(subplots=True)
# plt.show()
# plot_features = df[plot_cols][:480]
# plot_features.index = date_time[:480]
# _ = plot_features.plot(subplots=True)
# plt.show()

print(df.describe().transpose())

wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

print(df['wv (m/s)'].min())

# plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400)
# plt.colorbar()
# plt.xlabel('Wind Direction [deg]')
# plt.ylabel('Wind Velocity [m/s]')
# plt.show()

wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

wd_rad = df.pop('wd (deg)') * np.pi / 180

df['Wx'] = wv * np.cos(wd_rad)
df['Wy'] = wv * np.sin(wd_rad)

df['max Wx'] = max_wv * np.cos(wd_rad)
df['max Wy'] = max_wv * np.sin(wd_rad)

# plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
# plt.colorbar()
# plt.xlabel('Wind X [m/s]')
# plt.ylabel('Wind Y [m/x]')
# ax = plt.gca()
# ax.axis('tight')
# plt.show()

timestamp_s = date_time.map(datetime.datetime.timestamp)

day = 24 * 60 * 60
year = (365.2425) * day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

# plt.plot(np.array(df['Day sin'])[:25])
# plt.plot(np.array(df['Day cos'])[:25])
# plt.xlabel('Time [h]')
# plt.title('Time of day signal')
# plt.show()

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7): int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]
print(num_features)

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')

# plt.figure(figsize=(12, 6))
# ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
# _ = ax.set_xticklabels(df.keys(), rotation=90)
# plt.show()


class WindowGenerator():
    def __init__(
            self, input_width, label_width, shift,
            train_df=train_df, val_df=val_df, test_df=test_df,
            label_columns=None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indeices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indeices[name]] for name in self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indeices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.show()

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data, targets=None, sequence_length=self.total_window_size,
            sequence_stride=1, shuffle=True, batch_size=32
        )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result

        return result


w1 = WindowGenerator(input_width=24, label_width=1, shift=24, label_columns=['T (degC)'])
print(w1)
w2 = WindowGenerator(input_width=6, label_width=1, shift=1, label_columns=['T (degC)'])
print(w2)

example_windows = tf.stack([np.array(train_df[: w2.total_window_size]),
                            np.array(train_df[100:100+w2.total_window_size]),
                            np.array(train_df[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_windows)
print('All shapes are: (batch, time, features)')
print(f'Windows shape: {example_windows.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'labels shape: {example_labels.shape}')

# w2.example = example_inputs, example_labels
# w2.plot()

print(w2.train.element_spec)

for example_inputs, example_labels in w2.train.take(1):
    print(f'Input shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')

single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['T (degC)'])

print(single_step_window)

for example_inputs, example_labels in single_step_window.train.take(1):
    print(f'Input shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]

        return result[:, :, tf.newaxis]


val_performance = dict()
performance = dict()

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=['T (degC)']
)

MAX_EPOCHS = 20


def compile_and_fit(model, window, save_dir=None, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])

    if save_dir is not None:
        model.save(save_dir)

    return history


# baseline = Baseline(label_index=column_indices['T (degC)'])
# baseline.compile(loss=tf.losses.MeanSquaredError(),
#                  metrics=[tf.metrics.MeanAbsoluteError()])
# baseline.predict(single_step_window.val)
# baseline.save('data/time_series/baseline')
# baseline = tf.keras.models.load_model('data/time_series/baseline')
# val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
# performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
# print(val_performance)
# print(performance)
val_performance['Baseline'] = [0.012845649383962154, 0.07846629619598389]
performance['Baseline'] = [0.014162620529532433, 0.08516010642051697]
# wide_window.plot(baseline)


# linear = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=1)
# ])
# print(f'Input shape: {single_step_window.example[0].shape}')
# print(f'Output shape: {linear(single_step_window.example[0]).shape}')
# history = compile_and_fit(linear, single_step_window, 'data/time_series/linear')
# val_performance['Linear'] = linear.evaluate(single_step_window.val)
# performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)
# print(val_performance)
# print(performance)
val_performance['Linear'] = [0.008607517927885056, 0.06847935169935226]
performance['Linear'] = [0.008367066271603107, 0.06690479069948196]
# linear = tf.keras.models.load_model('data/time_series/linear')
# linear.summary()
# wide_window.plot(linear)
# plt.bar(x = range(len(train_df.columns)),
#         height=linear.layers[0].kernel[:,0].numpy())
# axis = plt.gca()
# axis.set_xticks(range(len(train_df.columns)))
# _ = axis.set_xticklabels(train_df.columns, rotation=90)
# plt.show()


# dense = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dense(units=1)
# ])
# history = compile_and_fit(dense, single_step_window, 'data/time_series/dense')
# val_performance['Dense'] = dense.evaluate(single_step_window.val)
# performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)
# print(val_performance)
# print(performance)
val_performance['Dense'] = [0.007389315869659185, 0.06134752556681633]
performance['Dense'] = [0.007670558523386717, 0.06355921924114227]
# dense = tf.keras.models.load_model('data/time_series/dense')
# wide_window.plot(dense)


CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['T (degC)']
)
# print()
# print(conv_window)
# conv_window.plot()


# multi_step_dense = tf.keras.Sequential([
#     # Shape: (time, features) => (time*features)
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=32, activation='relu'),
#     tf.keras.layers.Dense(units=32, activation='relu'),
#     tf.keras.layers.Dense(units=1),
#     # Shape: (outputs) => (1, outputs)
#     tf.keras.layers.Reshape([1, -1])
# ])
# print(f'Input shape: {conv_window.example[0].shape}')
# print(f'Output shape: {multi_step_dense(conv_window.example[0]).shape}')
# history = compile_and_fit(multi_step_dense, conv_window, 'data/time_series/multi_step_dense')
# val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
# performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)
# print(val_performance)
# print(performance)
val_performance['Multi step dense'] = [0.006903897970914841, 0.05890052020549774]
performance['Multi step dense'] = [0.00703806895762682, 0.06022002547979355]
# conv_window.plot(multi_step_dense)
# multi_step_dense = tf.keras.models.load_model('data/time_series/multi_step_dense')
# print('Input shape:', wide_window.example[0].shape)
# try:
#   print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
# except Exception as e:
#   print(f'\n{type(e).__name__}:{e}')

# conv_model = tf.keras.Sequential([
#     tf.keras.layers.Conv1D(filters=32, kernel_size=(CONV_WIDTH,), activation='relu'),
#     tf.keras.layers.Dense(units=32, activation='relu'),
#     tf.keras.layers.Dense(units=1)
# ])
# conv_model.build(input_shape=(32, 24, 19))
# conv_model.build(input_shape=(32, 3, 19))
# conv_model.summary()
# print('Conv model on "conv window"')
# print(f'Input shape: {conv_window.example[0].shape}')
# print(f'Output shape: {conv_model(conv_window.example[0]).shape}')
# history = compile_and_fit(conv_model, conv_window, 'data/time_series/conv')
# val_performance['Conv'] = conv_model.evaluate(conv_window.val)
# performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)
# print(val_performance)
# print(performance)
val_performance['Conv'] = [0.005989380646497011, 0.05319361388683319]
performance['Conv'] = [0.005987438373267651, 0.05417366698384285]
# conv_model = tf.keras.models.load_model('data/time_series/conv')
# print("Wide window")
# print('Input shape:', wide_window.example[0].shape)
# print('Labels shape:', wide_window.example[1].shape)
# print('Output shape:', conv_model(wide_window.example[0]).shape)
LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['T (degC)']
)
# print("Wide conv window")
# print('Input shape:', wide_conv_window.example[0].shape)
# print('Labels shape:', wide_conv_window.example[1].shape)
# print('Output shape:', conv_model(wide_conv_window.example[0]).shape)
# wide_conv_window.plot(conv_model)

# lstm_model = tf.keras.models.Sequential([
#     # Shape [batch, time, features] => [batch, time, lstm_units]
#     tf.keras.layers.LSTM(32, return_sequences=True),
#     # Shape => [batch, time, features]
#     tf.keras.layers.Dense(units=1)
# ])
# history = compile_and_fit(lstm_model, wide_window, 'data/time_series/lstm')
# val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
# performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)
# print(val_performance['LSTM'])
# print(performance['LSTM'])
val_performance['LSTM'] = [0.005590430926531553, 0.05157112330198288]
performance['LSTM'] = [0.005600317846983671, 0.052606042474508286]
# lstm_model = tf.keras.models.load_model('data/time_series/lstm')

# bilstm_model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv1D(filters=32, kernel_size=(3,), activation='relu', padding='same'),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True)),
#     tf.keras.layers.Dense(units=1)
# ])
# history = compile_and_fit(bilstm_model, wide_window, 'data/time_series/test_bilstm')
# val_performance['BiLSTM'] = bilstm_model.evaluate(wide_window.val)
# performance['BiLSTM'] = bilstm_model.evaluate(wide_window.test, verbose=0)
# print(val_performance['BiLSTM'])
# print(performance['BiLSTM'])
val_performance['BiLSTM'] = [0.0003349148319102824, 0.008535444736480713]
performance['BiLSTM'] = [0.000345767883118242, 0.009427332319319248]
# bilstm_model = tf.keras.models.load_model('data/time_series/bilstm')


# x = np.arange(len(performance))
# width = 0.3
# metric_name = 'mean_absolute_error'
# metric_index = bilstm_model.metrics_names.index('mean_absolute_error')
# val_mae = [v[metric_index] for v in val_performance.values()]
# test_mae = [v[metric_index] for v in performance.values()]
#
# plt.ylabel('mean_absolute_error [T (degC), normalized]')
# plt.bar(x - 0.17, val_mae, width, label='Validation')
# plt.bar(x + 0.17, test_mae, width, label='Test')
# plt.xticks(ticks=x, labels=performance.keys(),
#            rotation=45)
# _ = plt.legend()
# plt.show()

# for name, value in performance.items():
#   print(f'{name:12s}: {value[1]:0.4f}')


single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1)

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1)

val_performance = dict()
performance = dict()

# for example_inputs, example_labels in wide_window.train.take(1):
#   print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
#   print(f'Labels shape (batch, time, features): {example_labels.shape}')

# baseline = Baseline()
# baseline.compile(loss=tf.losses.MeanSquaredError(),
#                  metrics=[tf.metrics.MeanAbsoluteError()])
# val_performance['Baseline'] = baseline.evaluate(wide_window.val)
# performance['Baseline'] = baseline.evaluate(wide_window.test, verbose=0)
# print(val_performance)
# print(performance)
val_performance['Baseline'] = [0.0885537713766098, 0.15893250703811646]
performance['Baseline'] = [0.09026195108890533, 0.16375169157981873]

# dense = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dense(units=num_features)
# ])
# history = compile_and_fit(dense, single_step_window, 'data/time_series/mout_dense')
# val_performance['Dense'] = dense.evaluate(single_step_window.val)
# performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)
# print(val_performance['Dense'])
# print(performance['Dense'])
val_performance['Dense'] = [0.0670291930437088, 0.12778548896312714]
performance['Dense'] = [0.0676712915301323, 0.1289435178041458]

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1
)

# lstm_model = tf.keras.models.Sequential([
#     # Shape [batch, time,features] => [batch, time, lstm_units]
#     tf.keras.layers.LSTM(32, return_sequences=True),
#     # Shape => [batch, time, features]
#     tf.keras.layers.Dense(units=num_features)
# ])
# history = compile_and_fit(lstm_model, wide_window, 'data/time_series/mout_lstm')
# val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
# performance['LSTM'] = lstm_model.evaluate(wide_window.test)
# print(val_performance['LSTM'])
# print(performance['LSTM'])
val_performance['LSTM'] = [0.0616740919649601, 0.1203341856598854]
performance['LSTM'] = [0.06206468120217323, 0.12214010208845139]

# bilstm_model = tf.keras.models.Sequential([
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True)),
#     tf.keras.layers.Dense(units=num_features)
# ])
# history = compile_and_fit(bilstm_model, wide_window, 'data/time_series/mout_bilstm')
# val_performance['BiLSTM'] = bilstm_model.evaluate(wide_window.val)
# performance['BiLSTM'] = bilstm_model.evaluate(wide_window.test, verbose=0)
# print(val_performance['BiLSTM'])
# print(performance['BiLSTM'])
val_performance['BiLSTM'] = [0.003190754447132349, 0.017605125904083252]
performance['BiLSTM'] = [0.003205966902896762, 0.018013853579759598]


class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        return inputs + delta


residual_lstm = ResidualWrapper(
    tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(num_features, kernel_initializer=tf.initializers.zeros)
    ]))
residual_lstm.build(input_shape=(None, 24, 19))
history = compile_and_fit(residual_lstm, wide_window)
val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val)
performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test, verbose=0)
print(val_performance['Residual LSTM'])
print(performance['Residual LSTM'])

x = np.arange(len(performance))
width = 0.3

metric_name = 'mean_absolute_error'
metric_index = residual_lstm.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
plt.ylabel('MAE (average over all outputs)')
_ = plt.legend()
plt.show()

for name, value in performance.items():
  print(f'{name:15s}: {value[1]:0.4f}')