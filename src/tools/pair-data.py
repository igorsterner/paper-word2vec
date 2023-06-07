from pprint import pprint
import time
import nltk
from tqdm import tqdm
import csv
import pickle

SENTENCES = "data/eng_wikipedia_2016_1M/eng_wikipedia_2016_1M-sentences.txt"

print("Loading corpus...")
sentences = []
with open(
    SENTENCES,
    "r",
    encoding="utf-8",
) as f:
    corpus = csv.reader(f, delimiter="\t")

    for i, row in enumerate(corpus):
        sentences.append(row[1])

print("Finding all the words...")
words = []
for s in tqdm(sentences[:200000]):
    for w in nltk.word_tokenize(s):
        if w.isalpha():
            words.append(w)

word_context_pairs = []
print("Generating word-context pairs...")
for i, w in tqdm(enumerate(words[2:-2])):
    word_context_pairs.append((w, words[i]))
    word_context_pairs.append((w, words[i + 1]))
    word_context_pairs.append((w, words[i + 3]))
    word_context_pairs.append((w, words[i + 4]))

with open(f"data/word-context_pairs_200k_sents.pkl", "wb") as f:
    pickle.dump(word_context_pairs, f)
