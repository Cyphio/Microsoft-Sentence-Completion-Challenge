from QuestionFramework import Questions
from LanguageModel import LanguageModel
import os
from nltk import word_tokenize as tokenize
import numpy as np

class SentenceCompletionChallenge:
    def __init__(self):
        np.random.seed(101)

        num_training_files = 1
        self.lm = LanguageModel(num_training_files)

        self.questions = Questions().get_questions()

        self.choices = ["a", "b", "c", "d", "e"]


    def get_field(self, field):
        return [q.get_field(field) for q in self.questions]

    def predict(self, method="random"):
        # return [q.predict(method=method) for q in self.questions]
        if method == "random":
            return [self.choose_randomly in range(len(self.questions))]
        if method == "unigram":
            return [self.unigram(q=q) for q in self.questions]
        if method == "bigram":
            return [self.bigram(q) for q in self.questions]

    def predict_and_score(self, method="random"):
        scores = [int(p == a) for p, a in zip(self.predict(method=method), [q.answer for q in self.questions])]
        return sum(scores) / len(scores)

    def choose_randomly(self):
        return np.random.choice(self.choices)




    def unigram(self, q):
        probabilities = [self.lm.unigram.get(q.get_field(ch + ")"), 0) for ch in self.choices]
        best_choices = [ch for ch, prob in zip(self.choices, probabilities) if prob == max(probabilities)]
        # if len(best_choices)>1:
        #    print("Randomly choosing from {}".format(len(best_choices)))
        return np.random.choice(best_choices)



    def bigram(self, q, method="bigram"):
        context = self.get_left_context(q, window=1)
        probabilities = [self.lm.get_prob(q.get_field(f"{ch})"), context, methodparams={"method": method})
                         for ch in self.choices]
        best_choices = [ch for ch, prob in zip(self.choices, probabilities) if prob == max(probabilities)]
        # if len(best_choices)>1:
        #    print("Randomly choosing from {}".format(len(best_choices)))
        return np.random.choice(best_choices)

    def get_left_context(self, q, window, target='_____', sent_tokens=None):
        if sent_tokens is None:
            tokens = ["__START"] + tokenize(q.fields[q.col_names["question"]]) + ["__END"]
        else:
            tokens = sent_tokens
        if target in tokens:
            target_pos = tokens.index(target)
            if target_pos - window >= 0:
                return tokens[target_pos - window: target_pos]
        return []



if __name__ == '__main__':
    scc = SentenceCompletionChallenge()

    score = scc.predict_and_score("unigram")
    print(score)