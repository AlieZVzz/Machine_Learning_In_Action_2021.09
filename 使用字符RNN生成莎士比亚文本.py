import numpy as np
import tensorflow.keras.utils

shakespeare_url = "https://homl.info/shakespeare"
filepath = tensorflow.keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()

tokenizer = tensorflow.keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([shakespeare_text])

max_id = len(tokenizer.word_index)
dataset_size = tokenizer.document_count

[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
train_size = dataset_size * 90 // 100
dataset = tensorflow.data.Dataset.from_tensor_slices(encoded[:train_size])

n_steps = 100
window_length = n_steps + 1
dataset = dataset.window(window_length, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))

batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
dataset = dataset.map(lambda X_batch, y_batch: (tensorflow.one_hot(X_batch, depth=max_id), y_batch))

dataset = dataset.prefetch(1)

model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id], dropout=0.2,
                                      recurrent_dropout=0.2))
model.add(tensorflow.keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(tensorflow.keras.layers.TimeDistributed(tensorflow.keras.layers.Dense(max_id, activation='softmax')))

model.compile(loss=tensorflow.keras.losses.sparse_categorical_crossentropy, optimizer="adam")
history = model.fit(dataset, epochs=20)
