from QuestionFramework import Questions
from StatisticalLanguageModel import StatisticalLanguageModel
from LSTMNeuralLanguageModel import LSTMNeuralLanguageModel
import os
from nltk import word_tokenize as tokenize
import numpy as np
from pyprobar import probar


class SentenceCompletionChallenge():
    def __init__(self):
        np.random.seed(101)

        self.questions = Questions().get_questions()
        self.choices = ["a", "b", "c", "d", "e"]

    def get_field(self, field):
        return [q.get_field(field) for q in self.questions]

    def predict(self, methodparams=None):
        if methodparams is None:
            methodparams = {"model": "STATISTICAL", "n": 1, "smoothing": ""}

        if methodparams.get("model") == "STATISTICAL":
            self.statistical_lm = StatisticalLanguageModel(methodparams)
            return [self.statistical_pred(q=q, methodparams=methodparams) for q in self.questions]


        elif methodparams.get("model") == "NEURAL":
            self.neural_lm = LSTMNeuralLanguageModel(methodparams)
            model = self.neural_lm.load_model(model_path=methodparams.get("model_path"),
                                              model_data_path=methodparams.get("model_data_path"))
            print("GENERATING NEURAL LANGUAGE MODEL PREDICTIONS")
            return [self.neural_pred(q=q, model=model) for q in probar(self.questions)]


        else:
            print("Randomly predicting")
            return [self.choose_randomly in range(len(self.questions))]

    def predict_and_score(self, methodparams=None):
        scores = [int(p == a) for p, a in zip(self.predict(methodparams), [q.answer for q in self.questions])]
        return sum(scores) / len(scores)

    def choose_randomly(self):
        return np.random.choice(self.choices)

    def statistical_pred(self, q, methodparams):
        context = self.get_left_context(q)
        probabilities = [self.statistical_lm.get_prob(q.get_field(f"{ch})"), context, methodparams=methodparams)
                         for ch in self.choices]
        best_choices = [ch for ch, prob in zip(self.choices, probabilities) if prob == max(probabilities)]
        # if len(best_choices)>1:
        #    print("Randomly choosing from {}".format(len(best_choices)))
        return np.random.choice(best_choices)

    def neural_pred(self, q, model):
        context = ' '.join(self.get_left_context(q))
        # Removing punctuation from prime so that model doesnt run into unknown characters
        context = self.neural_lm.clean_text(context)
        # print(context)
        pred_word = self.neural_lm.get_pred_word(model, context, top_k=5)
        # print(f"ACTUAL: {q.get_field(f'a)')}")
        probabilities = [self.neural_lm.get_prob(pred_word, q.get_field(f"{ch})"))
                         for ch in self.choices]
        best_choices = [ch for ch, prob in zip(self.choices, probabilities) if prob == max(probabilities)]
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

    nueral_methodparams = {"model": "NEURAL",
                           "num_files": 1,
                           "model_path": "NEURAL_MODELS/LSTM/jolly-bush-44_epoch10.pth",
                           "model_data_path": "NEURAL_MODELS/LSTM/jolly-bush-44_data.csv"}

    score = scc.predict_and_score(nueral_methodparams)
    print(score)
