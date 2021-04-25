import os, math
from nltk import word_tokenize as tokenize
from nltk import ngrams
import operator
import numpy as np


class LanguageModel:
    def __init__(self, num_training_files, n):
        np.random.seed(101)

        train_data_set = "Holmes_data_set"
        self.training_files, self.held_out_files = self.get_training_testing(train_data_set)

        self.train_data_set = train_data_set
        self.files = self.training_files[:num_training_files]
        self.unigram = {}
        self.n_gram = {}
        # self.train()

    def get_training_testing(self, train_data_set, split=0.5):
        filenames = os.listdir(train_data_set)
        n = len(filenames)
        print("There are {} files in the training directory: {}".format(n, train_data_set))
        np.random.shuffle(filenames)
        index = int(n * split)
        return filenames[:index], filenames[index:]

    def train(self):
        self._processfiles()
        print(self.n_gram.items())
        self._make_unknowns()
        self._discount()
        self._convert_to_probs()

    def _processline(self, line):
        # tokens = ["__START"] + tokenize(line) + ["__END"]
        # previous = "__END"
        # for token in tokens:
        #     self.unigram[token] = self.unigram.get(token, 0) + 1
        #     current = self.n_gram.get(previous, {})
        #     current[token] = current.get(token, 0) + 1
        #     self.n_gram[previous] = current
        #     previous = token
        tokens = ["__START"] + tokenize(line) + ["__END"]
        self.n_gram = [' '.join(grams) for grams in ngrams(tokens, 2)]


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
        self.n_gram = {key: {k: v / sum(adict.values()) for (k, v) in adict.items()} for (key, adict) in
                       self.n_gram.items()}

    def get_prob(self, token, context="", methodparams=None):
        if methodparams is None:
            methodparams = {"method": "unigram", "smoothing": ""}
        # if method == "unigram":
        #     return self.unigram.get(token, 0)
        # elif method == "bigram":
        #     return self.bigram.get(context[-1], {}).get(token, 0)
        if methodparams.get("method") == "unigram":
            return self.unigram.get(token, self.unigram.get("__UNK", 0))
        else:
            if methodparams.get("smoothing") == "kneser-ney":
                uniform_distribution = self.kn
            else:
                uniform_distribution = self.unigram
            n_gram = self.n_gram.get(context[-1], self.n_gram.get("__UNK", {}))
            big_p = n_gram.get(token, n_gram.get("__UNK", 0))
            lmbda = n_gram["__DISCOUNT"]
            uniform_probability = uniform_distribution.get(token, uniform_distribution.get("__UNK", 0))
            # print(big_p,lmbda,uni_p)
            return big_p + lmbda * uniform_probability

    def gen_highly_probable_words(self, k, n):
        sorted_words = list(dict(sorted(self.unigram.items(), key=lambda item: item[1], reverse=True)).keys())[:k]
        return [np.random.choice(sorted_words) for i in range(n)]

    # use probabilities according to method to generate a likely next sequence
    # choose random token from k best
    def next_likely(self, k=1, current="", method="unigram"):
        blacklist = ["__START", "__UNK", "__DISCOUNT"]

        if method == "unigram":
            distribution = self.unigram
        else:
            distribution = self.n_gram.get(current, self.n_gram.get("__UNK", {}))

        most_likely = sorted(list(distribution.items()), key=operator.itemgetter(1), reverse=True)

        filtered = [w for (w, p) in most_likely if w not in blacklist]
        return np.random.choice(filtered[:k])

    def generate(self, k=1, end="__END", limit=20, methodparams=None):
        if methodparams is None:
            methodparams = {"method": "bigram"}
        # a very simplistic way of generating likely tokens according to the model
        current = "__START"
        tokens = []
        while current != end and len(tokens) < limit:
            current = self.next_likely(k=k, current=current, method=methodparams.get("method"))
            tokens.append(current)
        return " ".join(tokens[:-1])

    def compute_probability(self, filenames=None, method="unigram"):
        if filenames is None:
            filenames = self.files

        total_p, total_N = 0, 0
        for i, afile in enumerate(filenames):
            print("Processing file {}:{}".format(i, afile))
            try:
                with open(os.path.join(self.training_dir, afile)) as instream:
                    for line in instream:
                        line = line.rstrip()
                        if len(line) > 0:
                            p, N = self.compute_prob_line(line, method=method)
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

    def _make_unknowns(self, known=2):
        unknown = 0
        for (k, v) in list(self.unigram.items()):
            if v < known:
                del self.unigram[k]
                self.unigram["__UNK"] = self.unigram.get("__UNK", 0) + v
        for (k, adict) in list(self.n_gram.items()):
            for (kk, v) in list(adict.items()):
                isknown = self.unigram.get(kk, 0)
                if isknown == 0:
                    adict["__UNK"] = adict.get("__UNK", 0) + v
                    del adict[kk]
            isknown = self.unigram.get(k, 0)
            if isknown == 0:
                del self.n_gram[k]
                current = self.n_gram.get("__UNK", {})
                current.update(adict)
                self.n_gram["__UNK"] = current

            else:
                self.n_gram[k] = adict

    def _discount(self, discount=0.75):
        # discount each bigram count by a small fixed amount
        self.n_gram = {k: {kk: value - discount for (kk, value) in adict.items()} for (k, adict) in self.n_gram.items()}

        # for each word, store the total amount of the discount so that the total is the same
        # i.e., so we are reserving this as probability mass
        for k in self.n_gram.keys():
            lamb = len(self.n_gram[k])
            self.n_gram[k]["__DISCOUNT"] = lamb * discount

        # work out kneser-ney unigram probabilities
        # count the number of contexts each word has been seen in
        self.kn = {}
        for (k, adict) in self.n_gram.items():
            for kk in adict.keys():
                self.kn[kk] = self.kn.get(kk, 0) + 1


if __name__ == "__main__":
    num_training_files = 1
    lm = LanguageModel(num_training_files, 1)

    # print(lm.generate(k=5, methodparams={"method": "trigram"}))

    lm._processline("Hello there how are you")
    print(lm.unigram)
    print(lm.n_gram)
