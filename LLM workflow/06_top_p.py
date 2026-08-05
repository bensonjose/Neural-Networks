"""
STEP 6: TOP-P (NUCLEUS SAMPLING)
===================================
Instead of considering every possible word, keep only the smallest
group of top words whose probabilities add up to at least p (say,
0.9), and throw away the rest. This trims off the "long unlikely
tail" so the model can't pick something absurd.
"""

import numpy as np

vocab = ["cat", "dog", "mat", "on", "sat", "the"]
probs = np.array([0.10, 0.05, 0.30, 0.02, 0.13, 0.40])   # must sum to 1
print("Original probabilities:")
for word, p in zip(vocab, probs):
    print(f"  {word:5s} {p:.2f}")

def apply_top_p(probs, p):
    sorted_idx = np.argsort(probs)[::-1]      # highest probability first
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)

    cutoff = np.searchsorted(cumulative, p) + 1   # how many words to keep
    keep_idx = sorted_idx[:cutoff]

    filtered = np.zeros_like(probs)
    filtered[keep_idx] = probs[keep_idx]
    return filtered / filtered.sum(), keep_idx

for p_value in [0.5, 0.9]:
    filtered, kept = apply_top_p(probs, p_value)
    kept_words = [vocab[i] for i in kept]
    print(f"\ntop_p = {p_value}  -> keeps only: {kept_words}")
    for word, p in zip(vocab, filtered):
        if p > 0:
            print(f"  {word:5s} {p:.2f}")
