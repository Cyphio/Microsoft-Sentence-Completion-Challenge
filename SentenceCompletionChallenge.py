from QuestionFramework import Questions
from StatisticalLanguageModel import StatisticalLanguageModel
from RNNNeuralLanguageModel import RNNNeuralLanguageModel
import os
from nltk import word_tokenize as tokenize
import numpy as np


class SentenceCompletionChallenge:
    def __init__(self,):
        np.random.seed(101)

        self.questions = Questions().get_questions()
        self.choices = ["a", "b", "c", "d", "e"]

    def get_field(self, field):
        return [q.get_field(field) for q in self.questions]

    def predict(self, methodparams=None):
        if methodparams is None:
            methodparams = {"model": "STATISTICAL", "n": 1, "smoothing": ""}

        if methodparams.get("model") == "STATISTICAL":
            self.lm = StatisticalLanguageModel(methodparams)
        elif methodparams.get("model") == "NEURAL":
            self.lm = RNNNeuralLanguageModel(methodparams)
            if methodparams.get("test_model"):
                self.lm.load_model(input("PLEASE ENTER A MODEL PATH: "))
        else:
            print("Randomly predicting")
            return [self.choose_randomly in range(len(self.questions))]
        return [self.n_gram(q=q, methodparams=methodparams) for q in self.questions]

    def predict_and_score(self, methodparams=None):
        scores = [int(p == a) for p, a in zip(self.predict(methodparams), [q.answer for q in self.questions])]
        return sum(scores) / len(scores)

    def choose_randomly(self):
        return np.random.choice(self.choices)

    def n_gram(self, q, methodparams):
        context = self.get_left_context(q)
        probabilities = [self.lm.get_prob(q.get_field(f"{ch})"), context, methodparams=methodparams)
                         for ch in self.choices]
        best_choices = [ch for ch, prob in zip(self.choices, probabilities) if prob == max(probabilities)]
        # if len(best_choices)>1:
        #    print("Randomly choosing from {}".format(len(best_choices)))
        return np.random.choice(best_choices)

    def get_left_context(self, q, target='_____', sent_tokens=None):
        if sent_tokens is None:
            # tokens = ["__END", "__START"] + tokenize(q.fields[q.col_names["question"]]) + ["__END"]
            tokens = tokenize(q.fields[q.col_names["question"]])
        else:
            tokens = sent_tokens
        if target in tokens:
            target_pos = tokens.index(target)
            return tokens[:target_pos]
        return []


if __name__ == '__main__':
    scc = SentenceCompletionChallenge()

    n_gram_ann_methodparams = {"model": "NEURAL",
                               "num_files": 1,
                               "test_model": True}

    score = scc.predict_and_score(n_gram_ann_methodparams)
    print(score)
