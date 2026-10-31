# 4B - CBOW (Word2Vec) on Twitter dataset

import re
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# a. Data collection and preprocessing
data = pd.read_csv("twitter_dataset.csv")

def clean(text):
    text = text.lower()                          # lowercase
    text = re.sub(r"http\S+|@\w+|#", " ", text)  # remove links, @mentions, #
    text = re.sub("[^a-z ]", " ", text)          # keep only letters
    return [w for w in text.split() if w not in ENGLISH_STOP_WORDS]   # tokenize + remove stop words

tweets = data["Text"].apply(clean).tolist()
print("Example tweet tokens:", tweets[0])

# b. Train Word2Vec CBOW model (sg=0 means CBOW)
model = Word2Vec(tweets, vector_size=100, window=5, min_count=2, sg=0, epochs=10)
print("Vocabulary size:", len(model.wv))

# c. Evaluate results - check learned embeddings with examples
words = model.wv.index_to_key[:5]     # 5 most common words
print("Common words:", words)

w = words[0]
print("\nEmbedding of '" + w + "' (first 10 values):", model.wv[w][:10])

for w in words:
    print("\nMost similar to '" + w + "':", model.wv.most_similar(w, topn=5))

print("\nSimilarity between", words[0], "and", words[1], ":", model.wv.similarity(words[0], words[1]))
