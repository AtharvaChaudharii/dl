# 4D - Recommendation System using CBOW word embeddings (Twitter dataset)

import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# 1. LOAD
data = pd.read_csv("twitter_dataset.csv")
texts = data["Text"].tolist()

# 2. CLEAN (lowercase, keep letters, tokenize, remove stop words)
def clean(text):
    text = re.sub("[^a-z ]", " ", text.lower())
    return [w for w in text.split() if w not in ENGLISH_STOP_WORDS]

sentences = [clean(t) for t in texts]

# 3. TRAIN WORD2VEC CBOW (sg=0 means CBOW)
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0, epochs=10)
print("Vocabulary size:", len(w2v.wv))

# 4. DOCUMENT VECTOR = average of all its word embeddings
def doc_vector(words):
    words = [w for w in words if w in w2v.wv]                # ignore unknown words typed by user
    return np.mean([w2v.wv[w] for w in words], axis=0) if words else np.zeros(100)

x = np.array([doc_vector(s) for s in sentences])

# 5. RECOMMEND - cosine similarity between user input and every tweet
query = doc_vector(clean(input("Enter keywords (e.g. 'sell economy development'): ")))
scores = cosine_similarity([query], x)[0]
for i in scores.argsort()[::-1][:5]:                          # top 5, highest first
    print(round(scores[i], 3), "->", texts[i][:100].replace("\n", " "))
