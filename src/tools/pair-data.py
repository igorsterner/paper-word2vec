from pprint import pprint
import time
import nltk
from tqdm import tqdm
import csv
import pickle

import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter

log_path = "logging"

logger = logging.getLogger(__name__)

from main import initialization

SENTENCES = "data/eng_wikipedia_2016_1M/eng_wikipedia_2016_1M-sentences.txt"

logger.info("Loading corpus...")
sentences = []
with open(
    SENTENCES,
    "r",
    encoding="utf-8",
) as f:
    corpus = csv.reader(f, delimiter="\t")

    for i, row in enumerate(corpus):
        sentences.append(row[1])

logger.info("Finding all the words...")
words = []
for s in tqdm(sentences[:200000]):
    for w in nltk.word_tokenize(s):
        if w.isalpha():
            words.append(w)

word_context_pairs = []
logger.info("Generating word-context pairs...")
for i, w in tqdm(enumerate(words[2:-2])):
    word_context_pairs.append((w, words[i]))
    word_context_pairs.append((w, words[i + 1]))
    word_context_pairs.append((w, words[i + 3]))
    word_context_pairs.append((w, words[i + 4]))

with open(f"data/word-context_pairs_200k_sents.pkl", "wb") as f:
    pickle.dump(word_context_pairs, f)
