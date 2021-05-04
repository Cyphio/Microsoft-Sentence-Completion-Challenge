import os, math
from nltk import word_tokenize as tokenize
from nltk import ngrams
import operator
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import deque
import csv
import pandas as pd


class StatisticalLanguageModel:
    def __init__(self, params):
        np.random.seed(101)

        train_data_set = "Holmes_data_set"
        self.training_files, self.held_out_files = self.get_training_testing(train_data_set)

        self.train_data_set = train_data_set
        self.files = self.training_files[:params.get("num_files")]

        self.params = params

        self.uni_gram = {}
        self.bi_gram = {}
        self.tri_gram = {}

        self._process_files()
        self._make_unknowns()
        self._discount()
        self._convert_to_probs()

    def get_training_testing(self, train_data_set, split=0.5):
        filenames = os.listdir(train_data_set)
        n = len(filenames)
        print("There are {} files in the training directory: {}".format(n, train_data_set))
        np.random.shuffle(filenames)
        index = int(n * split)
        return filenames[:index], filenames[index:]

    def _process_line(self, line):
        tokens = ["__START"] + tokenize(line) + ["__END"]
        previous = "__END"

        for token in tokens:
            # Creating uni_gram
            self.uni_gram[token] = self.uni_gram.get(token, 0) + 1

            # Creating bi_gram
            bi_current = self.bi_gram.get(previous, {})
            bi_current[token] = bi_current.get(token, 0) + 1
            self.bi_gram[previous] = bi_current
            previous = token

        # Creating tri_gram
        shifted_tokens = deque(tokens)
        shifted_tokens.rotate(1)
        for prev_token, token in zip(shifted_tokens, tokens):
            tri_current = self.tri_gram.get(previous, {})
            tri_current[token] = tri_current.get(token, 0) + 1
            self.tri_gram[previous] = tri_current
            previous = (prev_token, token)

    def _process_files(self):
        for file in self.files:
            print("Processing {}".format(file))
            try:
                with open(os.path.join(self.train_data_set, file)) as in_stream:
                    for line in in_stream:
                        line = line.rstrip()
                        if len(line) > 0:
                            self._process_line(line)
            except UnicodeDecodeError:
                print("UnicodeDecodeError processing {}: ignoring file".format(file))

    def _convert_to_probs(self):
        self.uni_gram = {k: v / sum(self.uni_gram.values()) for (k, v) in self.uni_gram.items()}
        self.bi_gram = {key: {k: v / sum(adict.values()) for (k, v) in adict.items()} for (key, adict) in
                        self.bi_gram.items()}
        self.tri_gram = {key: {k: v / sum(adict.values()) for (k, v) in adict.items()} for (key, adict) in
                         self.tri_gram.items()}

    def get_prob(self, token, context=None):

        # bi_gram
        if self.params.get("n") == 2:
            bi_gram = self.bi_gram.get(context[-1], self.bi_gram.get("__UNK", {}))
            big_p = bi_gram.get(token, bi_gram.get("__UNK", 0))
            if self.params.get("smoothing") == "absolute":
                uni_dist = self.uni_gram
            elif self.params.get("smoothing") == "kneser-ney":
                uni_dist = self.bigram_kn
            else:
                # No smoothing
                return big_p
            # Probability of token occurring after given context (big_p) + proportion of reserved probability mass
            # (lambda) according to the uni_gram probability of the token (if absolute smoothing) or according to
            # the likelihood of the token being seen in novel word combinations (if kneser-ney)
            return big_p + (bi_gram["__DISCOUNT"] * uni_dist.get(token, uni_dist.get("__UNK", 0)))

        # tri_gram
        elif self.params.get("n") == 3:
            tri_gram = self.tri_gram.get(tuple(context[-2:]), self.tri_gram.get("__UNK", {}))
            big_p = tri_gram.get(token, tri_gram.get("__UNK", 0))
            if self.params.get("smoothing") == "absolute":
                uni_dist = self.uni_gram
            elif self.params.get("smoothing") == "kneser-ney":
                uni_dist = self.trigram_kn
            else:
                # No smoothing
                return big_p
            return big_p + (tri_gram["__DISCOUNT"] * uni_dist.get(token, uni_dist.get("__UNK", 0)))

        # uni_gram
        else:
            print("PREDICTING USING UNI_GRAM")
            return self.uni_gram.get(token, self.uni_gram.get("__UNK", 0))

    def gen_highly_probable_words(self, k, n):
        sorted_words = list(dict(sorted(self.uni_gram.items(), key=lambda item: item[1], reverse=True)).keys())[:k]
        return [np.random.choice(sorted_words) for i in range(n)]

    # use probabilities according to method to generate a likely next sequence
    # choose random token from k best
    def next_likely(self, n, k=1, current=""):
        blacklist = ["__START", "__UNK", "__DISCOUNT"]

        if n == 3:
            distribution = self.tri_gram.get(current, self.tri_gram.get("__UNK", {}))
        elif n == 2:
            distribution = self.bi_gram.get(current, self.bi_gram.get("__UNK", {}))
        else:
            distribution = self.uni_gram

        most_likely = sorted(list(distribution.items()), key=operator.itemgetter(1), reverse=True)

        filtered = [w for (w, p) in most_likely if w not in blacklist]
        return np.random.choice(filtered[:k])

    def generate(self, k=1, end="__END", limit=20):
        # a very simplistic way of generating likely tokens according to the model
        current = "__START"
        tokens = []
        while current != end and len(tokens) < limit:
            current = self.next_likely(n=self.params.get("n"), k=k, current=current)
            tokens.append(current)
        return " ".join(tokens[:-1])

    def compute_prob_line(self, line):
        # this will add _start to the beginning of a line of text
        # compute the probability of the line according to the desired model
        # and returns probability together with number of tokens

        tokens = ["__END", "__START"] + tokenize(line) + ["__END"]
        acc = 0

        for i, token in enumerate(tokens[1:]):
            acc += math.log(
                self.get_prob(token, tokens[:i + 1]))
        return acc, len(tokens[1:])

    def compute_probability(self):
        if filenames is None:
            filenames = self.files

        total_p, total_N = 0, 0
        for i, file in enumerate(self.files):
            print("Processing file {}: {}".format(i, file))
            try:
                with open(os.path.join(self.train_data_set, file)) as in_stream:
                    for line in in_stream:
                        line = line.rstrip()
                        if len(line) > 0:
                            p, N = self.compute_prob_line(line=line)
                            total_p += p
                            total_N += N
            except UnicodeDecodeError:
                print("UnicodeDecodeError processing file {}: ignoring rest of file".format(file))
        return total_p, total_N

    # compute the probability and length of the corpus
    # calculate perplexity
    # lower perplexity means that the model better explains the data
    def compute_perplexity(self):
        p, N = self.compute_probability()
        # print(p,N)
        pp = math.exp(-p / N)
        return pp

    def _make_unknowns(self, known=2):
        unknown = 0
        for (k, v) in list(self.uni_gram.items()):
            if v < known:
                del self.uni_gram[k]
                self.uni_gram["__UNK"] = self.uni_gram.get("__UNK", 0) + v

        for (k, adict) in list(self.bi_gram.items()):
            for (kk, v) in list(adict.items()):
                isknown = self.uni_gram.get(kk, 0)
                if isknown == 0:
                    adict["__UNK"] = adict.get("__UNK", 0) + v
                    del adict[kk]
            isknown = self.uni_gram.get(k, 0)
            if isknown == 0:
                del self.bi_gram[k]
                current = self.bi_gram.get("__UNK", {})
                current.update(adict)
                self.bi_gram["__UNK"] = current
            else:
                self.bi_gram[k] = adict

        for (k, adict) in list(self.tri_gram.items()):
            for (kk, v) in list(adict.items()):
                isknown = self.uni_gram.get(kk, 0)
                if isknown == 0:
                    adict["__UNK"] = adict.get("__UNK", 0) + v
                    del adict[kk]

            known = [self.uni_gram.get(token, 0) for token in k]
            if 0 in known:
                del self.tri_gram[k]
                current = self.tri_gram.get("__UNK", {})
                current.update(adict)
                self.tri_gram["__UNK"] = current
            else:
                self.tri_gram[k] = adict

            # known = [self.uni_gram.get(token, 0) for token in k]
            # for idx, token in enumerate(k):
            #     if known[idx] == 0:
            #         lst = list(k)
            #         lst[idx] = "__UNK"
            #         k = tuple(lst)
            #         current = self.tri_gram.get("__UNK", {})
            #         current.update(adict)
            #         self.tri_gram["__UNK"] = current
            #     else:
            #         self.tri_gram[k] = adict

            # print(k)
            # for idx, token in enumerate(k):
            #     isknown = self.uni_gram.get(token, 0)
            #     if isknown == 0:
            #         del self.tri_gram[k]
            #         current = self.tri_gram.get("__UNK", {})
            #         current.update(adict)
            #         self.tri_gram["__UNK"] = current
            #         x = list(k)
            #         x[idx] = "__UNK"
            #         k = tuple(x)
            #     else:
            #         self.tri_gram[k] = adict

    def _discount(self, discount=0.75):
        # discount each bigram count by a small fixed amount
        self.bi_gram = {k: {kk: value - discount for (kk, value) in adict.items()} for (k, adict) in
                        self.bi_gram.items()}
        self.tri_gram = {k: {kk: value - discount for (kk, value) in adict.items()} for (k, adict) in
                         self.tri_gram.items()}

        # for each word, store the total amount of the discount so that the total is the same
        # i.e., so we are reserving this as probability mass
        for k in self.bi_gram.keys():
            lamb = len(self.bi_gram[k])
            self.bi_gram[k]["__DISCOUNT"] = lamb * discount
            # print(self.bi_gram[k])
        for k in self.tri_gram.keys():
            lamb = len(self.tri_gram[k])
            self.tri_gram[k]["__DISCOUNT"] = lamb * discount
            # print(self.tri_gram[k])

        # work out kneser-ney unigram probabilities
        # count the number of contexts each word has been seen in
        self.bigram_kn = {}
        for (bigram_k, bigram_dict) in self.bi_gram.items():
            for bigram_kk in bigram_dict.keys():
                self.bigram_kn[bigram_kk] = self.bigram_kn.get(bigram_kk, 0) + 1
        # print(self.bigram_kn)

        # self.trigram_kn = {}
        # for (trigram_k, tigram_dict) in self.tri_gram.items():
        #     for trigram_kk, bigram_dict in zip(tigram_dict.keys(), self.bi_gram.values()):
        #         # for bigram_dict in self.bi_gram.values():
        #             if trigram_kk in bigram_dict.keys():
        #                 self.trigram_kn[trigram_kk] = self.trigram_kn.get(trigram_kk, 0) + 1
        # print(self.bi_gram)
        # print(self.trigram_kn)

        self.trigram_kn = {}
        for (trigram_k, trigram_dict) in self.tri_gram.items():
            for trigram_kk in trigram_dict.keys():
                self.trigram_kn[trigram_kk] = self.trigram_kn.get(trigram_kk, 0) + 1
        # print(self.trigram_kn)


if __name__ == "__main__":
    params = {"model": "STATISTICAL",
              "num_files": 1,
              "n": 3,
              "smoothing": "kneser-ney"}
    lm = StatisticalLanguageModel(params)

    prob = lm.get_prob(token="this", context="had done")
    print(prob)
