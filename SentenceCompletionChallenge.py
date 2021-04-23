import pandas as pd, csv
import os
from nltk import word_tokenize as tokenize
import numpy as np

class SentenceCompletionChallenge:
    def __init__(self):
        np.random.seed(101)

        # PARAMETERS
        self.left_context_window = 2

        self.question_path = "testing_data.csv"
        self.answer_path = "test_answer.csv"

        self.read_files()

    def read_files(self):
        with open(self.question_path) as instream:
            csvreader = csv.reader(instream)
            qlines = list(csvreader)

        q_colnames = {item: i for i, item in enumerate(qlines[0])}
        self.questions = [question(qline) for qline in qlines[1:]]

        self.q_df = pd.DataFrame(qlines[1:], columns=qlines[0])
        self.q_df['tokens'] = self.q_df['question'].map(tokenize)
        self.q_df['left_context'] = self.q_df['tokens'].map(lambda x: self.get_left_context(x, self.left_context_window))

        with open(self.answer_path) as instream:
            csvreader = csv.reader(instream)
            alines = list(csvreader)

        for q, aline in zip(self.questions, alines[1:]):
            q.add_answer(aline)

    def get_left_context(self, sent_tokens, window, target='_____'):
        if target in sent_tokens:
            target_pos = sent_tokens.index(target)
            if target_pos - window >= 0:
                return sent_tokens[target_pos - window: target_pos]
        return []

    def get_field(self, field):
        return [q.get_field(field) for q in self.questions]

    def predict(self, method="random"):
        return [q.predict(method=method) for q in self.questions]

    def predict_and_score(self, method="random"):
        scores = [q.predict_and_score(method=method) for q in self.questions]
        return sum(scores) / len(scores)


class question:
    def __init__(self, aline):
        self.fields = aline

    def get_field(self, field):
        return self.fields[question.colnames[field]]

    def add_answer(self, fields):
        self.answer = fields[1]

    def predict_and_score(self, method="chooseA"):
        return int(self.predict(method=method) == self.answer)

    def predict(self, method="random"):
        if method == "random":
            return self.choose_randomly()

    def choose_randomly(self):
        return np.random.choice(["a", "b", "c", "d", "e"])



if __name__ == '__main__':
    scc = SentenceCompletionChallenge()

    score = scc.predict_and_score()
    print(score)