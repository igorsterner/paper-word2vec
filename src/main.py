import pickle
from collections import Counter, defaultdict
from pprint import pprint
import os
import sys

import numpy as np
from numpy import dot
from numpy.linalg import norm, svd
from scipy.sparse.linalg import svds
from tqdm import tqdm

import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter

log_path = "logging"

logger = logging.getLogger(__name__)

max_freq = 400

debug = False


class word2vec:
    def __init__(self) -> None:
        pass

    def load_data(self):
        word_context_pairs = []
        for i in tqdm(range(1, 20)):
            with open(f"data/word-context_pairs_{i}000000.pkl", "rb") as f:
                word_context_pairs += pickle.load(f)

        # Use lin dependency data
        pos_data = []
        for i in tqdm(range(1, 3)):
            with open(f"data/dependencies_{i}00000_pos.pkl", "rb") as f:
                pos_data += pickle.load(f)
        # word_context_pairs = [(t[0], t[3]) for t in pos_data]

        logger.info(f"Number of word-context pairs: {len(word_context_pairs)}")

        noun_vocab = []
        for trip in pos_data:
            if trip[1] == "NOUN":
                noun_vocab.append(trip[0])

        c = Counter(noun_vocab)
        noun_vocab = [w for w, freq in c.items() if freq > max_freq]

        logging.info("Generating vocab lists")

        word_vocab = []
        cont_vocab = []

        for trip in tqdm(word_context_pairs):
            word_vocab.append(trip[0])
            cont_vocab.append(trip[1])

        c = Counter(word_vocab)
        word_vocab = [w for w, freq in c.items() if freq > max_freq]

        c = Counter(cont_vocab)
        cont_vocab = [w for w, freq in c.items() if freq > max_freq]

        if debug:
            logging.info(f"Length of word vocab = {len(word_vocab)}")
            logging.info(f"Length of context vocab = {len(cont_vocab)}")

        logging.info("Generating PMI matrix data")
        logging.info(f"Length of word vocab = {len(word_vocab)}")
        logging.info(len(set(noun_vocab).intersection(word_vocab)))

        noun_vocab = set(noun_vocab).intersection(word_vocab)

        num_word_context = defaultdict(int)
        num_word = defaultdict(int)
        num_context = defaultdict(int)

        for pair in tqdm(word_context_pairs):
            num_word_context[pair] += 1
            num_word[pair[0]] += 1
            num_context[pair[1]] += 1

        num_pairs = len(word_context_pairs)
        word_context_pairs = []

        self.noun_vocab = noun_vocab
        self.num_word_context = num_word_context
        self.num_word = num_word
        self.num_context = num_context

    def SPPMI(self, k):
        logging.info("Generating PPMI matrix...")
        pmi_matrix = []
        for word in tqdm(self.noun_vocab):
            row = []
            for context in self.noun_vocab:
                pmi = np.log(
                    (self.num_word_context[(word, context)] * self.num_pairs)
                    / (self.num_word[word] * self.num_context[context])
                )
                row.append(max(0, pmi - np.log(k)))
            pmi_matrix.append(row)

        pmi_matrix = np.matrix([*pmi_matrix])

        # print(pmi_matrix.shape)
        # print("Generating thesarus...")
        # output = {}
        # for i, word in enumerate(tqdm(self.noun_vocab)):
        #     theas = {}
        #     for j, comp in enumerate(self.noun_vocab):
        #         if word == comp:
        #             continue
        #         theas[comp] = pmi_matrix[i, j]
        #     total = sum(theas.values())
        #     theas = {k: v/total for k, v in theas.items()}
        #     output[word] = list(sorted(theas.items(), key=lambda item: item[1], reverse=True))

        # with open(f'sppmi-dep/results_k{k}.pkl', 'wb') as f:
        #     pickle.dump(output, f)

        print("Calculating SVD...")
        Ud, Sigma, Vd = svds(pmi_matrix, k=d, which="LM")
        # Ud, Sigma, Vd = svd(pmi_matrix, full_matrices=True)
        if debug:
            print(f"PMI matrix shape: {pmi_matrix.shape}")
            print(f"Ud matrix shape: {Ud.shape}")
            print(f"Vd.T matrix shape: {(Vd.T).shape}")

        sigma_diag = np.diag(Sigma)
        sigma_12 = np.sqrt(sigma_diag)

        W = Ud @ sigma_12
        C = Vd.T @ sigma_12

        print("Generating thesarus...")
        output = {}
        for i, word in enumerate(tqdm(self.noun_vocab)):
            word_embed = W[i]
            theas = {}
            for j, comp in enumerate(self.noun_vocab):
                if word == comp:
                    continue
                comp_embed = W[j]
                theas[comp] = float(
                    (word_embed @ comp_embed.T) / (norm(word_embed) * norm(comp_embed))
                )
            output[word] = list(
                sorted(theas.items(), key=lambda item: item[1], reverse=True)
            )[:100]

        with open(f"thesauri/svd_noun/results_d{d}_k{k}.pkl", "wb") as f:
            pickle.dump(output, f)


def initialization():
    # ====== Set Logger =====
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s : %(message)s (in %(pathname)s:%(lineno)d)"
    log_console_format = "[%(levelname)s] - %(name)s : %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))
    from utils.color_logging import CustomFormatter

    custom_output_formatter = CustomFormatter(custom_format=log_console_format)
    console_handler.setFormatter(custom_output_formatter)

    info_file_handler = RotatingFileHandler(
        os.path.join(log_path, "info.log"),
        maxBytes=10**6,
        backupCount=5,
    )

    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(Formatter(log_file_format))

    exp_file_handler = RotatingFileHandler(
        os.path.join(log_path, "debug.log"),
        maxBytes=10**6,
        backupCount=5,
    )

    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler(
        os.path.join(log_path, "error.log"),
        maxBytes=10**6,
        backupCount=5,
    )
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(info_file_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)

    # setup a hook to log unhandled exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        logger.error(
            f"Uncaught exception: {exc_type} --> {exc_value}",
            exc_info=(exc_type, exc_value, exc_traceback),
        )


if __name__ == "__main__":
    ds = [100, 200, 300]
    ks = [1, 2, 3, 4, 5, 10]

    W2V = word2vec()

    W2V.load_data()

    for d in tqdm(ds):
        for k in ks:
            logger.info(f"d = {d}, k = {k}:")
            W2V.SPPMI(k)
