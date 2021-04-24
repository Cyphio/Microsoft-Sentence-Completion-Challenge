import os, random, math
from nltk import word_tokenize as tokenize
import operator
import numpy as np


class LanguageModel:
    def __init__(self, num_training_files):
        train_data_set = "Holmes_data_set"
        self.training_files, self.held_out_files = self.get_training_testing(train_data_set)

        self.train_data_set = train_data_set
        self.files = self.training_files[:num_training_files]
        self.unigram = {}
        self.bigram = {}
        self.train()

    def get_training_testing(self, train_data_set, split=0.5):
        filenames = os.listdir(train_data_set)
        n = len(filenames)
        print("There are {} files in the training directory: {}".format(n, train_data_set))
        np.random.shuffle(filenames)
        index = int(n * split)
        return filenames[:index], filenames[index:]

    def train(self):
        self._processfiles()
        self._convert_to_probs()

    def _processline(self, line):
        tokens = ["_START"] + tokenize(line) + ["_END"]
        previous = "__END"
        for token in tokens:
            self.unigram[token] = self.unigram.get(token, 0) + 1
            current = self.bigram.get(previous, {})
            current[token] = current.get(token, 0) + 1
            self.bigram[previous] = current
            previous = token

    def _processfiles(self):
        for afile in self.files:
            print("Processing {}".format(afile))
            try:
                with open(os.path.join(self.train_data_set, afile)) as instream:
                    for line in instream:
                        line = line.rstrip()
                        if len(line) > 0:
                            self._processline(line)
            except UnicodeDecodeError:
                print("UnicodeDecodeError processing {}: ignoring file".format(afile))

    def _convert_to_probs(self):
        self.unigram = {k: v / sum(self.unigram.values()) for (k, v) in self.unigram.items()}
        self.bigram = {key: {k: v / sum(adict.values()) for (k, v) in adict.items()} for (key, adict) in
                       self.bigram.items()}

    def get_prob(self, token, method="unigram"):
        if method == "unigram":
            return self.unigram.get(token, 0)
        elif method == "bigram":
            return self.bigram.get(context[-1], {}).get(token, 0)

    def gen_highly_probable_words(self, k, n):
        sorted_words = list(dict(sorted(self.unigram.items(), key=lambda item: item[1], reverse=True)).keys())[:k]
        return [np.random.choice(sorted_words) for i in range(n)]

    # use probabilities according to method to generate a likely next sequence
    # choose random token from k best
    def next_likely(self, k=1, current="", method="unigram"):
        blacklist = ["__START"]

        if method == "unigram":
            dist = self.unigram
        else:
            dist = self.bigram.get(current, {})

        # sort the tokens by unigram probability
        mostlikely = sorted(list(dist.items()), key=operator.itemgetter(1), reverse=True)

        # filter out any undesirable tokens
        filtered = [w for (w, p) in mostlikely if w not in blacklist]
        # choose one randomly from the top k

        res = random.choice(filtered[:k])
        return res

    def generate(self, k=1, end="__END", limit=20):
        # a very simplistic way of generating likely tokens according to the model
        current = "__START"
        tokens = []
        while current != end and len(tokens) < limit:
            current = self.nextlikely(k=k, current=current, method=method)
            tokens.append(current)
        return " ".join(tokens[:-1])

    def compute_probability(self, filenames=None, method="unigram"):
        if filenames is None:
            filenames = self.files

        total_p, total_N = 0, 0
        for i,afile in enumerate(filenames):
            print("Processing file {}:{}".format(i,afile))
            try:
                with open(os.path.join(self.training_dir,afile)) as instream:
                    for line in instream:
                        line=line.rstrip()
                        if len(line) > 0:
                            p, N = self.compute_prob_line(line,method=method)
                            total_p += p
                            total_N += N
            except UnicodeDecodeError:
                print("UnicodeDecodeError processing file {}: ignoring rest of file".format(afile))
        return total_p, total_N

    # compute the probability and length of the corpus
    # calculate perplexity
    # lower perplexity means that the model better explains the data
    def compute_perplexity(self, filenames=None, method="unigram"):
        if filenames is None:
            filenames = []
        p, N = self.compute_probability(filenames=filenames, method=method)
        # print(p,N)
        pp = math.exp(-p / N)
        return pp
