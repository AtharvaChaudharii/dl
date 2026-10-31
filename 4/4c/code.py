# 4C - CBOW (Word2Vec) on document3.txt

import re
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# a. Data preparation
text = open("document3.txt", encoding="utf-8").read().lower()

sentences = []
for s in text.split("."):                    # split into sentences
    s = re.sub("[^a-z0-9 ]", " ", s)         # remove punctuation
    words = [w for w in s.split() if w not in ENGLISH_STOP_WORDS]
    if words:
        sentences.append(words)

print("Sentences:", sentences)

# b. Train Word2Vec CBOW model (sg=0 means CBOW)
# small document -> min_count=1 and more epochs
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=0, epochs=200)
print("Vocabulary:", model.wv.index_to_key)

# c. Evaluate results
print("\nEmbedding of 'influenza' (first 10 values):", model.wv["influenza"][:10])
print("\nMost similar to 'influenza':", model.wv.most_similar("influenza", topn=5))
print("\nMost similar to 'transmission':", model.wv.most_similar("transmission", topn=5))
print("\nMost similar to 'virus':", model.wv.most_similar("virus", topn=5))
print("\nSimilarity influenza vs covid:", model.wv.similarity("influenza", "covid"))
