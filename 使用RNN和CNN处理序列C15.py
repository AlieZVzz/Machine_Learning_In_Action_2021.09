import numpy as np
import tensorflow.keras.models


def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)


n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000, -1]

import tensorflow.keras as Keras

y_pred = X_valid[:, -1]
np.mean(Keras.losses.mean_squared_error(y_valid, y_pred))

model = tensorflow.keras.models.Sequential()
model.add(Keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]))
model.add(Keras.layers.SimpleRNN(20, return_sequences=True))
model.add(Keras.layers.Dense(1))

checkpoint_cb = tensorflow.keras.callbacks.ModelCheckpoint("RNN_model.h5")

model.compile(optimizer='rmsprop', loss=Keras.losses.mean_squared_error, metrics=['mse'])
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10, batch_size=128, callbacks=[checkpoint_cb])
