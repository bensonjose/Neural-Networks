"""
STEP 1: TOKENIZATION
=====================
Turning text into numbers. A model can't read words -- it reads
integers. So every unique word gets assigned an ID.
"""

text = "the cat sat on the mat"

# split text into words
words = text.split()
print("Words:", words)

# give every unique word a number
vocab = sorted(set(words))
word_to_id = {word: i for i, word in enumerate(vocab)}
print("Vocabulary (word -> id):", word_to_id)

# convert the sentence into token ids
token_ids = [word_to_id[w] for w in words]
print("Token IDs:", token_ids)

# convert back (id -> word), to prove it's reversible
id_to_word = {i: w for w, i in word_to_id.items()}
decoded = " ".join(id_to_word[i] for i in token_ids)
print("Decoded back to text:", decoded)
