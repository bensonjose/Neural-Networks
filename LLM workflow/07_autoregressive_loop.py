"""
STEP 7: AUTOREGRESSIVE LOOP
==============================
Generate ONE token, add it to the sequence, then feed the WHOLE
sequence back in to predict the NEXT token. Repeat. This is why
generation happens one word at a time, left to right.

Here we fake "next word prediction" with a simple hand-made rule
just to keep focus on the LOOP mechanic itself, not the model.
"""

import numpy as np
np.random.seed(1)

# a tiny hand-made "knowledge" of what tends to follow what
# (in a real model, this comes from steps 1-6 combined)
next_word_options = {
    "the":  ["cat", "dog"],
    "cat":  ["sat"],
    "dog":  ["sat"],
    "sat":  ["on"],
    "on":   ["the"],
}

def predict_next_word(sequence):
    last_word = sequence[-1]
    options = next_word_options.get(last_word, ["the"])
    return np.random.choice(options)   # the "sampling" step

def generate(prompt, num_steps):
    sequence = prompt.split()
    print(f"Start: {sequence}")

    for step in range(num_steps):
        next_word = predict_next_word(sequence)     # predict
        sequence.append(next_word)                  # append
        print(f"  step {step+1}: sequence is now -> {' '.join(sequence)}")
        # loop repeats: the WHOLE sequence (including next_word) goes
        # back in as input for the next prediction

    return " ".join(sequence)

final_text = generate("the cat", num_steps=6)
print(f"\nFinal text: {final_text!r}")
