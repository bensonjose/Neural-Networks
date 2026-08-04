"""
STEP 5: TEMPERATURE
=====================
Temperature reshapes the probability distribution BEFORE sampling.
  - LOW temperature (e.g. 0.3)  -> sharper, more confident, repetitive
  - HIGH temperature (e.g. 2.0) -> flatter, more random, more "creative"
  - temperature = 1.0 means "no change"
"""

import numpy as np

vocab = ["cat", "dog", "mat", "on", "sat", "the"]
logits = np.array([1.2, 0.3, 2.1, -0.5, 0.8, 3.0])

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def apply_temperature(logits, temperature):
    return softmax(logits / temperature)

for temp in [0.3, 1.0, 2.0]:
    probs = apply_temperature(logits, temp)
    print(f"\nTemperature = {temp}")
    for word, prob in zip(vocab, probs):
        bar = "#" * int(prob * 40)
        print(f"  {word:5s} {prob:.3f}  {bar}")
