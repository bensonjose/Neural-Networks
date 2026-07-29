# Neural-Networks
Simple Explanation of How a Basic Neural Network Works:
Also a Basic LLM workflow
Hope it's easy to understand!!

# LLM Workflow

THis is a simple set of programs that show how a language model works internally, step by step. Each file demonstrates one concept in isolation.
run them in order (01 -> 07) to build up the full picture.


---

## The Steps

### 1. `01_tokenization.py` — Text → Numbers
Models can't read words, only numbers. This splits a sentence into words and assigns each unique word an ID.

### 2. `02_embeddings.py` — Numbers → Meaning
Each word ID gets converted into a vector (a list of numbers) meant to capture its "meaning." Similar words end up with similar vectors.

### 3. `03_transformer_attention.py` — Understanding Context
The core of a transformer: for every word, decide how much attention to pay to every earlier word, then blend that info together. This is how a model uses context (e.g. knowing "it" refers to "the cat" mentioned earlier).

### 4. `04_probability_distribution.py` — Scoring Every Possible Next Word
The model outputs one raw score per vocabulary word. Softmax turns those raw scores into real probabilities that add up to 1.

### 5. `05_temperature.py` — Controlling Randomness
Reshapes the probability distribution before picking a word:
- **Low temperature** → confident, predictable, repetitive
- **High temperature** → more random, more "creative"

### 6. `06_top_p.py` — Cutting Off Bad Options
Instead of considering every word, keep only the smallest group of top words whose probabilities add up to `p` (e.g. 0.9) — this trims away unlikely, nonsensical choices.

### 7. `07_autoregressive_loop.py` — Generating Text, One Word at a Time
Predict one word, add it to the sentence, then feed the WHOLE sentence back in to predict the next word. Repeat. This is why generation happens left-to-right, one token at a time.

---


These scripts use tiny toy data and random or hand-made weights instead of a real trained model, the goal is to see the **mechanics** clearly, not to produce smart text. A real LLM uses the exact same steps, just at a much larger scale, with weights learned from huge amounts of real text.