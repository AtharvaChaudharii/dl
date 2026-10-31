# 4A - CBOW (Word2Vec) for Document Classification on IMDB reviews

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from tensorflow import keras

# 1. Data preparation
data = pd.read_csv("IMDB Dataset.csv")

def clean(text):
    text = text.lower()                          # lowercase
    text = re.sub("<.*?>", " ", text)            # remove html like <br />
    text = re.sub("[^a-z ]", " ", text)          # keep only letters
    return [w for w in text.split() if w not in ENGLISH_STOP_WORDS]   # tokenize + remove stop words

sentences = data["review"].apply(clean).tolist()              # list of sentences (list of words)
labels = (data["sentiment"] == "positive").astype(int).values  # positive = 1, negative = 0

x_train_words, x_test_words, y_train, y_test = train_test_split(sentences, labels, test_size=0.2)

# 2 & 3. Train Word2Vec with CBOW (sg=0) on training data -> builds the vocabulary + embeddings
w2v = Word2Vec(x_train_words, vector_size=100, window=5, min_count=2, sg=0)
print("Vocabulary size:", len(w2v.wv))

# Document representation = average of all its word embeddings
def doc_vector(words):
    vecs = [w2v.wv[w] for w in words if w in w2v.wv]
    if len(vecs) == 0:
        return np.zeros(100)
    return np.mean(vecs, axis=0)

x_train = np.array([doc_vector(d) for d in x_train_words])
x_test = np.array([doc_vector(d) for d in x_test_words])

# Feed forward neural network for classification
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(100,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")      # positive / negative
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

h = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 4. Evaluate on test set
loss, acc = model.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

plt.plot(h.history["accuracy"], label="train acc")
plt.plot(h.history["val_accuracy"], label="val acc")
plt.plot(h.history["loss"], label="train loss")
plt.plot(h.history["val_loss"], label="val loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()
