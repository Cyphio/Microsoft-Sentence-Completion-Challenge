import pandas as pd, csv
import os
from nltk import word_tokenize as tokenize

class SentenceCompletionChallenge:
    def __init__(self, parentdir):
        # PARAMETERS
        self.left_context_window = 2

        self.question_path = os.path.join(parentdir, "testing_data.csv")
        self.answer_path = os.path.join(parentdir, "test_answer.csv")

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

    def predict(self, method="chooseA"):
        return [q.predict(method=method) for q in self.questions]

    def predict_and_score(self, method="chooseA"):
        scores = [q.predict_and_score(method=method) for q in self.questions]
        return sum(scores) / len(scores)


class question:

    def __init__(self, aline):
        self.fields = aline

    def get_field(self, field):
        return self.fields[question.colnames[field]]

    def add_answer(self, fields):
        self.answer = fields[1]

    def chooseA(self):
        return ("a")

    def predict(self, method="chooseA"):
        # eventually there will be lots of methods to choose from
        if method == "chooseA":
            return self.chooseA()

    def predict_and_score(self, method="chooseA"):
        # compare prediction according to method with the correct answer
        # return 1 or 0 accordingly
        return int(self.predict(method=method) == self.answer)