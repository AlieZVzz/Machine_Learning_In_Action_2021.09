import tensorflow.keras.datasets.imdb

import tensorflow_datasets as tfds
import tensorflow as tf
from collections import Counter

(X_train, y_train), (X_test, y_test) = tensorflow.keras.datasets.imdb.load_data()
word_index = tensorflow.keras.datasets.imdb.get_word_index()
id_to_word = {id_ + 3: word for word, id_ in word_index.items()}
for id_, token in enumerate(('<pad>', "<sos>", "<unk>")):
    id_to_word[id_] = token
" ".join([id_to_word[id_] for id_ in X_train[0][:10]])

datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_size = info.splits['train'].num_examples


def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b"")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b"")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch


vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    for review in X_batch:
        vocabulary.update(list(review.numpy()))

vocab_size = 10000
truncated_vocabulary = [word for word, couont in vocabulary.most_common()[:vocab_size]]

words = tf.constant(truncated_vocabulary)
words_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, words_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)


def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch


train_set = datasets["train"].batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)

embed_size = 128
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size, input_shape=[None]))
model.add(tensorflow.keras.layers.GRU(128, return_sequences=True))
model.add(tensorflow.keras.layers.GRU(128))
model.add(tensorflow.keras.layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_set, epochs=5)
