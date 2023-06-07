import spacy
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

# gpu = spacy.prefer_gpu()
# print(gpu)

logger.info("Loading NLP model...")
nlp = spacy.load("en_core_web_sm")

logger.info("Loading corpus...")
sentences = []
with open(
    "data/eng_wikipedia_2016_1M/eng_wikipedia_2016_1M-sentences.txt",
    "r",
    encoding="utf-8",
) as f:
    corpus = csv.reader(f, delimiter="\t")

    for i, row in enumerate(corpus):
        sentences.append(row[1])

dependency_triples = []

logger.info("Parsing sentences...")

start = time.time()

for i, sentence in tqdm(enumerate(sentences)):
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == "ROOT" or token.dep_ == "punct":
            continue
        dependency_triples.append(
            (
                token.text,
                token.pos_,
                f"{token.dep_}-of",
                token.head.text,
                token.head.pos_,
            )
        )

    if int(i + 1) % 100000 == 0:
        with open(f"data/dependencies_{str(i+1)}_pos.pkl", "wb") as f:
            pickle.dump(dependency_triples, f)

        dependency_triples = []


end = time.time()

logger.info(end - start)
logger.info(len(dependency_triples))

if dependency_triples:
    with open(f"data/dependencies_remaining.pkl", "wb") as f:
        pickle.dump(dependency_triples, f)

# pprint(dependency_tripes)
