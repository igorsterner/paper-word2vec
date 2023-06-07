import pickle
from pprint import pprint
import argparse
import os

import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter

log_path = "logging"

logger = logging.getLogger(__name__)

from main import initialization

with open("thesauri/svd_noun/results_d300_k1.pkl", "rb") as f:
    theasarus = pickle.load(f)


def rnns():
    rnns = {}
    for word1 in theasarus.keys():
        for word2 in theasarus.keys():
            if word1 == word2:
                continue
            elif (word2, word1) in rnns:
                continue

            word1_theas = theasarus[word1][0][0]
            word2_theas = theasarus[word2][0][0]

            if word1 == word2_theas and word2 == word1_theas:
                rnns[(word1, word2)] = round(theasarus[word1][0][1], 2)

    pprint(sorted(rnns.items(), key=lambda item: item[1], reverse=True))


def write_theasarus():
    with open("theasarus.txt", "w", encoding="utf-8") as f:
        for x in theasarus.keys():
            f.write(x)
            f.write("\n")


def query_theasarus(word, sim=False):
    top_10 = theasarus[word][:10]
    total = sum(x[1] for x in top_10)

    if sim:
        return [(w[0], round(w[1] / total, 2)) for w in top_10]
    else:
        return [w[0] for w in top_10]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--word")
    parser.add_argument("--find_rnns")
    args = parser.parse_args()

    initialization()

    if args.word:
        logging.info(f"Original word: {args.word}")
        pprint(query_theasarus(args.word, sim=True))
    elif args.find_rnns:
        logging.info("Finding RNNs like the paper")
        rnns()

    # write_theasarus()
