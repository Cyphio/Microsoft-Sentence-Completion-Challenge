from QuestionFramework import Questions
from LanguageModel import LanguageModel
import os
from nltk import word_tokenize as tokenize
import numpy as np


class SentenceCompletionChallenge:
    def __init__(self, num_training_files, save_lm=False):
        np.random.seed(101)

        self.lm = LanguageModel(num_training_files, save_data=save_lm)

        self.questions = Questions().get_questions()

        self.choices = ["a", "b", "c", "d", "e"]


    def get_field(self, field):
        return [q.get_field(field) for q in self.questions]

    def predict(self, method="random", smoothing=""):
        # return [q.predict(method=method) for q in self.questions]
        if method == "uni_gram":
            return [self.uni_gram(q=q) for q in self.questions]
        if method == "bi_gram" or "tri_gram":
            return [self.n_gram(q=q, method=method, smoothing=smoothing) for q in self.questions]
        # Else predict randomly
        print("Randomly predicting")
        return [self.choose_randomly in range(len(self.questions))]

    def predict_and_score(self, method="random", smoothing=""):
        scores = [int(p == a) for p, a in zip(self.predict(method, smoothing), [q.answer for q in self.questions])]
        return sum(scores) / len(scores)




    def choose_randomly(self):
        return np.random.choice(self.choices)



    def uni_gram(self, q):
        probabilities = [self.lm.uni_gram.get(q.get_field(ch + ")"), 0) for ch in self.choices]
        best_choices = [ch for ch, prob in zip(self.choices, probabilities) if prob == max(probabilities)]
        # if len(best_choices)>1:
        #    print("Randomly choosing from {}".format(len(best_choices)))
        return np.random.choice(best_choices)



    def n_gram(self, q, method, smoothing=""):
        context = self.get_left_context(q)
        probabilities = [self.lm.get_prob(q.get_field(f"{ch})"), context, methodparams={"method": method,
                                                                                        "smoothing": smoothing})
                         for ch in self.choices]
        best_choices = [ch for ch, prob in zip(self.choices, probabilities) if prob == max(probabilities)]
        # if len(best_choices)>1:
        #    print("Randomly choosing from {}".format(len(best_choices)))
        return np.random.choice(best_choices)

    def get_left_context(self, q, target='_____', sent_tokens=None):
        if sent_tokens is None:
            tokens = ["__END", "__START"] + tokenize(q.fields[q.col_names["question"]]) + ["__END"]
        else:
            tokens = sent_tokens
        if target in tokens:
            target_pos = tokens.index(target)
            return tokens[:target_pos]
        return []



if __name__ == '__main__':
    num_training_files = 10
    scc = SentenceCompletionChallenge(num_training_files, save_lm=False)
    score = scc.predict_and_score(method="tri_gram", smoothing="kneser-ney")
    print(score)