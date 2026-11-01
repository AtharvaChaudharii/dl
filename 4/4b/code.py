# 4B - CBOW (Word2Vec) on Twitter dataset

import re
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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

# 4. EVALUATE - check embeddings with examples
words = w2v.wv.index_to_key[:5]                      # 5 most common words
print("Embedding of", words[0], ":", w2v.wv[words[0]][:10])
for w in words:
    print("Most similar to", w, ":", w2v.wv.most_similar(w, topn=5))
print("Similarity", words[0], words[1], ":", w2v.wv.similarity(words[0], words[1]))
