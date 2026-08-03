"""
STEP 4: PROBABILITY DISTRIBUTION (SOFTMAX)
=============================================
The model's final layer outputs one raw "score" (logit) per word
in the vocabulary. Softmax turns those raw scores into proper
probabilities that all add up to 1 -- so we can treat "how likely
is each next word" as an actual probability.
"""

import numpy as np

vocab = ["cat", "dog", "mat", "on", "sat", "the"]

# pretend these are raw scores the model produced for "what comes next"
# (higher = model thinks it's more likely)
logits = np.array([1.2, 0.3, 2.1, -0.5, 0.8, 3.0])

def softmax(x):
    x = x - np.max(x)          # for numerical stability, doesn't change result
    e = np.exp(x)
    return e / np.sum(e)

probs = softmax(logits)

print("Word    Logit   Probability")
for word, logit, prob in zip(vocab, logits, probs):
    print(f"{word:6s}  {logit:5.2f}   {prob:.3f}")

print(f"\nProbabilities sum to: {probs.sum():.3f}  (should be 1.0)")
print(f"Most likely next word: '{vocab[np.argmax(probs)]}'")
