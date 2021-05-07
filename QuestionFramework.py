from StatisticalLanguageModel import StatisticalLanguageModel
import os
import numpy as np
import pandas as pd, csv
from nltk import word_tokenize as tokenize


class Questions:
    def __init__(self, question_path="testing_data.csv", answer_path="test_answer.csv"):

        with open(question_path) as instream:
            csvreader = csv.reader(instream)
            qlines = list(csvreader)

        Question.col_names = {item: i for i, item in enumerate(qlines[0])}

        self.questions = [Question(qline) for qline in qlines[1:]]

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