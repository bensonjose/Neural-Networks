"""
STEP 2: EMBEDDINGS
===================
Turning a token ID into a vector of numbers (a list of floats).
This vector is meant to capture "meaning" -- similar words end up
with similar vectors. Here we just use random numbers to show the
MECHANICS (a real model learns these values from data).
"""

import numpy as np
np.random.seed(0)

vocab = ["cat", "dog", "mat", "on", "sat", "the"]
vocab_size = len(vocab)
embed_dim = 4   # how many numbers represent each word

# one row per word -- this IS the embedding table
embedding_table = np.random.randn(vocab_size, embed_dim).round(2)

print("Embedding table (one row of numbers per word):")
for word, vector in zip(vocab, embedding_table):
    print(f"  {word:5s} -> {vector}")

# look up the embedding for a specific word
word = "cat"
word_id = vocab.index(word)
vector = embedding_table[word_id]
print(f"\nEmbedding lookup for '{word}': {vector}")

# look up embeddings for a whole sentence
sentence = ["the", "cat", "sat"]
sentence_ids = [vocab.index(w) for w in sentence]
sentence_vectors = embedding_table[sentence_ids]
print(f"\nEmbeddings for sentence {sentence}:")
print(sentence_vectors)
