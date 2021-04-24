from LanguageModel import LanguageModel
import os
import numpy as np
import pandas as pd, csv


class Questions:
    def __init__(self, question_path="testing_data.csv", answer_path="test_answer.csv"):

        with open(question_path) as instream:
            csvreader = csv.reader(instream)
            qlines = list(csvreader)

        Question.col_names = {item: i for i, item in enumerate(qlines[0])}

        self.questions = [Question(qline) for qline in qlines[1:]]

        # self.q_df = pd.DataFrame(qlines[1:], columns=qlines[0])
        # self.q_df['tokens'] = self.q_df['question'].map(tokenize)
        # self.q_df['left_context'] = self.q_df['tokens'].map(lambda x: self.get_left_context(x, self.left_context_window))

        with open(answer_path) as instream:
            csvreader = csv.reader(instream)
            alines = list(csvreader)

        for q, aline in zip(self.questions, alines[1:]):
            q.add_answer(aline)

    def get_questions(self):
        return self.questions




class Question:
    def __init__(self, aline):
        self.fields = aline

    def get_field(self, field):
        return self.fields[Question.col_names[field]]

    def add_answer(self, fields):
        self.answer = fields[1]

    # def predict_and_score(self, method="chooseA"):
    #     return int(self.predict(method=method) == self.answer)
    #
    # def predict(self, method="random", language_model=self.lm):
    #     if method == "random":
    #         return self.choose_randomly()
    #     if method == "unigram":
    #         return self.choose_unigram(language_model)
    #     if method == "bigram":
    #         return self.choose_bigram(language_model)


