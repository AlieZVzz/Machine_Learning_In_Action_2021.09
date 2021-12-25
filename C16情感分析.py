import tensorflow

(X_train, y_train), (X_test, y_test) = tensorflow.keras.datasets.imdb.load_data()
word_index = tensorflow.keras.datasets.imdb.get_word_index()
id_to_word = {id_ + 3: word for word, id_ in word_index.items()}
for id_, token in enumerate(('<pad>', "<sos>","<unk>")):
    id_to_word[id_]=token
" ".join([id_to_word[id_] for id_ in X_train[0][:10]])