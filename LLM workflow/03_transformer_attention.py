"""
STEP 3: TRANSFORMER (SELF-ATTENTION)
======================================
The core trick: for every word, figure out how much attention to
pay to every OTHER word in the sentence, then blend their info
together. This is what lets a model understand context.

Example: in "the cat sat on the mat", when processing "sat", the
model can look back at "cat" to know WHO is sitting.
"""

import numpy as np
np.random.seed(0)

sentence = ["the", "cat", "sat", "on", "the", "mat"]
seq_len = len(sentence)
embed_dim = 4

# pretend embeddings (normally step 2's output)
x = np.random.randn(seq_len, embed_dim).round(2)

# Query, Key, Value weight matrices (random here, learned in real models)
W_q = np.random.randn(embed_dim, embed_dim) * 0.5
W_k = np.random.randn(embed_dim, embed_dim) * 0.5
W_v = np.random.randn(embed_dim, embed_dim) * 0.5

Q = x @ W_q   # "what am I looking for?"      (per word)
K = x @ W_k   # "what do I offer?"            (per word)
V = x @ W_v   # "what info do I actually carry?"

# how much does each word match every other word?
scores = Q @ K.T / np.sqrt(embed_dim)

# CAUSAL MASK: block each word from seeing words that come AFTER it
mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
scores = scores + mask

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

attention_weights = softmax(scores)

print("Attention weights -- each row = how much that word looks at every earlier word:\n")
print("        " + "  ".join(f"{w:>5s}" for w in sentence))
for i, word in enumerate(sentence):
    row = "  ".join(f"{v:5.2f}" for v in attention_weights[i])
    print(f"{word:6s}  {row}")

# blend the V vectors according to those attention weights
output = attention_weights @ V
print("\nFinal blended output for each word position (shape):", output.shape)
print("(each word's vector is now a mix of itself + earlier context)")
