# 4A - CBOW (Word2Vec) for Document Classification on IMDB reviews

import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from tensorflow import keras

# 1. LOAD
data = pd.read_csv("IMDB Dataset.csv")
texts = data["review"].tolist()
y = (data["sentiment"] == "positive").astype(int).values      # positive = 1, negative = 0

# 2. CLEAN (lowercase, keep letters, tokenize, remove stop words)
def clean(text):
    text = re.sub("[^a-z ]", " ", text.lower())
    return [w for w in text.split() if w not in ENGLISH_STOP_WORDS]

sentences = [clean(t) for t in texts]

# 3. TRAIN WORD2VEC CBOW (sg=0 means CBOW) -> vocabulary + word embeddings
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0, epochs=10)
print("Vocabulary size:", len(w2v.wv))

# 4. DOCUMENT VECTOR = average of all its word embeddings
def doc_vector(words):
    words = [w for w in words if w in w2v.wv]                # ignore unknown words typed by user
    return np.mean([w2v.wv[w] for w in words], axis=0) if words else np.zeros(100)

x = np.array([doc_vector(s) for s in sentences])

# 5. CLASSIFY with a feed forward neural network
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(100,)),
    keras.layers.Dense(1, activation="sigmoid")              # positive / negative
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10)

loss, acc = model.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", acc)
