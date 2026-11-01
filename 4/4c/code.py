# 4C - CBOW (Word2Vec) on document3.txt

import re
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 1. LOAD (split the document into sentences)
texts = open("document3.txt", encoding="utf-8").read().split(".")

# 2. CLEAN (lowercase, keep letters, tokenize, remove stop words)
def clean(text):
    text = re.sub("[^a-z ]", " ", text.lower())
    return [w for w in text.split() if w not in ENGLISH_STOP_WORDS]

sentences = [clean(t) for t in texts]

# 3. TRAIN WORD2VEC CBOW (sg=0 means CBOW)
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0, epochs=200)   # tiny text -> many epochs
print("Vocabulary size:", len(w2v.wv))

# 4. EVALUATE - check embeddings with examples
words = w2v.wv.index_to_key[:5]                      # 5 most common words
print("Embedding of", words[0], ":", w2v.wv[words[0]][:10])
for w in words:
    print("Most similar to", w, ":", w2v.wv.most_similar(w, topn=5))
print("Similarity", words[0], words[1], ":", w2v.wv.similarity(words[0], words[1]))
