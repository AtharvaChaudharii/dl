# 4D - Recommendation System using CBOW word embeddings (Twitter dataset)

import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# 1. Data collection and preprocessing
data = pd.read_csv("twitter_dataset.csv")

def clean(text):
    text = text.lower()                          # lowercase
    text = re.sub(r"http\S+|@\w+|#", " ", text)  # remove links, @mentions, #
    text = re.sub("[^a-z ]", " ", text)          # keep only letters
    return [w for w in text.split() if w not in ENGLISH_STOP_WORDS]   # tokenize + remove stop words

docs = data["Text"].apply(clean).tolist()

# 2a. Train Word2Vec CBOW model (sg=0 means CBOW)
model = Word2Vec(docs, vector_size=100, window=5, min_count=2, sg=0, epochs=10)

# 2b/2d. Document vector = average of embeddings of all its words
def doc_vector(words):
    vecs = [model.wv[w] for w in words if w in model.wv]
    if len(vecs) == 0:
        return np.zeros(100)
    return np.mean(vecs, axis=0)

doc_vectors = np.array([doc_vector(d) for d in docs])

# 2c/3. Recommendation: cosine similarity between user input and every document
def recommend(user_input, top_n=5):
    query = doc_vector(clean(user_input)).reshape(1, -1)
    scores = cosine_similarity(query, doc_vectors)[0]
    best = scores.argsort()[::-1][:top_n]          # highest similarity first
    for i in best:
        print(round(scores[i], 3), "->", data["Text"][i][:100].replace("\n", " "))

user_input = input("Enter keywords (e.g. 'sell economy development'): ")
recommend(user_input)
