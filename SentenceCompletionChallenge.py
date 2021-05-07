from QuestionFramework import Questions
from StatisticalLanguageModel import StatisticalLanguageModel
from NeuralLanguageModel import NeuralLanguageModel
import os
from nltk import word_tokenize as tokenize
import numpy as np
from pyprobar import probar
import matplotlib.pyplot as plt


class SentenceCompletionChallenge:
    def __init__(self):
        np.random.seed(101)

        self.questions = Questions().get_questions()
        self.choices = ["a", "b", "c", "d", "e"]

        self.pred_count = 0
        self.choice_count = 0
        self.num_random_choice_count = 0

    def get_field(self, field):
        return [q.get_field(field) for q in self.questions]

    def predict(self, params=None):
        if params is None:
            params = {"model": "STATISTICAL", "n": 1, "smoothing": ""}

        if params.get("model") == "STATISTICAL":
            self.statistical_lm = StatisticalLanguageModel(params)
            print("GENERATING STATISTICAL LANGUAGE MODEL PREDICTIONS")
            preds = [self.statistical_pred(q=q) for q in probar(self.questions)]
            print(f"AVG NUM OF CHOICES: {self.choice_count/self.pred_count}\n"
                  f"RANDOMLY CHOOSING {(self.num_random_choice_count/self.pred_count)*100}% OF THE TIME")
            return preds


        elif params.get("model") == "NEURAL":
            self.neural_lm = NeuralLanguageModel(params)
            model = self.neural_lm.load_model(model_path=params.get("model_path"),
                                              model_data_path=params.get("model_data_path"))
            print("GENERATING NEURAL LANGUAGE MODEL PREDICTIONS")
            preds = [self.neural_pred(q=q, model=model) for q in probar(self.questions)]
            print(f"AVG NUM OF CHOICES: {self.choice_count/self.pred_count}\n"
                  f"RANDOMLY CHOOSING {(self.num_random_choice_count/self.pred_count)*100}% OF THE TIME")
            return preds


        else:
            print("Randomly predicting")
            return [self.choose_randomly in range(len(self.questions))]

    def predict_and_score(self, methodparams=None):
        scores = [int(p == a) for p, a in zip(self.predict(methodparams), [q.answer for q in self.questions])]
        return sum(scores) / len(scores)

    def choose_randomly(self):
        return np.random.choice(self.choices)

    def statistical_pred(self, q):
        context = self.get_left_context(q)
        probabilities = [self.statistical_lm.get_prob(q.get_field(f"{ch})"), context)
                         for ch in self.choices]
        best_choices = [ch for ch, prob in zip(self.choices, probabilities) if prob == max(probabilities)]
        self.choice_count += len(best_choices)
        if len(best_choices) > 1:
            self.num_random_choice_count += 1
        self.pred_count += 1
        return np.random.choice(best_choices)

    def neural_pred(self, q, model):
        context = ' '.join(self.get_left_context(q))
        # Removing punctuation from prime so that model doesnt run into unknown characters
        context = self.neural_lm._clean_text(context)
        pred_word = self.neural_lm.get_pred_word(model, context, top_k=5)
        probabilities = [self.neural_lm.get_prob(pred_word, q.get_field(f"{ch})"))
                         for ch in self.choices]
        best_choices = [ch for ch, prob in zip(self.choices, probabilities) if prob == max(probabilities)]
        self.choice_count += len(best_choices)
        if len(best_choices) > 1:
            self.num_random_choice_count += 1
        self.pred_count += 1
        return np.random.choice(best_choices)

    def get_left_context(self, q, target='_____', sent_tokens=None):
        if sent_tokens is None:
            tokens = tokenize(q.fields[q.col_names["question"]])
        else:
            tokens = sent_tokens
        if target in tokens:
            target_pos = tokens.index(target)
            return tokens[:target_pos]
        return []

    def get_scc_accuracy_bar_chart(self, param_list, save_data=False):
        labels = []
        accuracies = []
        for idx, param in enumerate(param_list):
            score = self.predict_and_score(param)
            print(f"IDX: {idx}, SCORE: {score}")
            accuracies.append(score)
            model = param.get('model')
            num_files = param.get('num_files')
            if model == "STATISTICAL":
                labels.append(
                    f"{model}, smoothing: {param.get('smoothing')}\nn: {param.get('n')}, num files: {num_files}")
            else:
                labels.append(f"{model}\nnum files: {num_files}")
        plt.bar(np.arange(len(labels)), accuracies, alpha=0.5)
        plt.xticks(np.arange(len(labels)), labels, rotation=90)
        plt.ylabel('Accuracy')
        plt.title('Bar chart of SCC accuracy across different Language Models')
        plt.tight_layout()
        if save_data:
            plt.savefig(f"acc_barchart.png")
        plt.show()

    def get_ngram_acc_perp_scatter(self, param_list, save_data=False):
        accuracies = []
        perplexities = []
        for param in param_list:
            accuracies.append(self.predict_and_score(param))
            lm = StatisticalLanguageModel(param)
            perplexities.append(lm.compute_perplexity())
        plt.scatter(accuracies, perplexities)
        plt.xlabel("Accuracy")
        plt.ylabel("Perplexity")
        plt.title("Accuracy against perplexity for Statistical Language Models")
        plt.tight_layout()
        if save_data:
            plt.savefig("scatter_plot.png")
        plt.show()

if __name__ == '__main__':
    scc = SentenceCompletionChallenge()

    neural_params = {"model": "NEURAL",
                     "num_files": None,
                     "model_path": "NEURAL_MODELS/LSTM/jolly-bush-44_epoch50.pth",
                     "model_data_path": "NEURAL_MODELS/LSTM/jolly-bush-44_data.csv"}

    stat_params = {"model": "STATISTICAL",
                   "num_files": 50,
                   "n": 2,
                   "smoothing": "kneser-ney"}

    score = scc.predict_and_score(stat_params)
    print(f"SCORE: {score}")

    # model = "STATISTICAL"
    # num_files = 50
    # smoothing = "kneser-ney"
    # scc.get_scc_accuracy_bar_chart([{"model": model,
    #                                          "num_files": num_files,
    #                                          "n": 1,
    #                                          "smoothing": None},
    #                                         {"model": model,
    #                                          "num_files": num_files,
    #                                          "n": 2,
    #                                          "smoothing": smoothing},
    #                                         {"model": model,
    #                                          "num_files": num_files,
    #                                          "n": 3,
    #                                          "smoothing": smoothing}], save_data=True)

    # scc.get_scc_accuracy_bar_chart([{"model": model,
    #                                  "num_files": num_files,
    #                                  "n": 1,
    #                                  "smoothing": None},
    #                                 {"model": model,
    #                                  "num_files": num_files,
    #                                  "n": 2,
    #                                  "smoothing": None},
    #                                 {"model": model,
    #                                  "num_files": num_files,
    #                                  "n": 3,
    #                                  "smoothing": None},
    #                                 {"model": model,
    #                                  "num_files": num_files,
    #                                  "n": 2,
    #                                  "smoothing": "absolute"},
    #                                 {"model": model,
    #                                  "num_files": num_files,
    #                                  "n": 3,
    #                                  "smoothing": "absolute"},
    #                                 {"model": model,
    #                                  "num_files": num_files,
    #                                  "n": 2,
    #                                  "smoothing": "kneser-ney"},
    #                                 {"model": model,
    #                                  "num_files": num_files,
    #                                  "n": 3,
    #                                  "smoothing": "kneser-ney"},
    #                                 {"model": "NEURAL",
    #                                  "num_files": 50,
    #                                  "model_path": "NEURAL_MODELS/LSTM/misty-monkey-50_epoch50.pth",
    #                                  "model_data_path": "NEURAL_MODELS/LSTM/misty-monkey-50_data.csv"},
    #                                 {"model": "NEURAL",
    #                                  "num_files": 250,
    #                                  "model_path": "NEURAL_MODELS/LSTM/jolly-bush-44_epoch50.pth",
    #                                  "model_data_path": "NEURAL_MODELS/LSTM/jolly-bush-44_data.csv"}
    #                                 ], save_data=True)

    # scc.get_ngram_acc_perp_scatter([{"model": model,
    #                                  "num_files": num_files,
    #                                  "n": 1,
    #                                  "smoothing": None},
    #                                 {"model": model,
    #                                  "num_files": num_files,
    #                                  "n": 2,
    #                                  "smoothing": None},
    #                                 {"model": model,
    #                                  "num_files": num_files,
    #                                  "n": 3,
    #                                  "smoothing": None},
    #                                 {"model": model,
    #                                  "num_files": num_files,
    #                                  "n": 2,
    #                                  "smoothing": "absolute"},
    #                                 {"model": model,
    #                                  "num_files": num_files,
    #                                  "n": 3,
    #                                  "smoothing": "absolute"},
    #                                 {"model": model,
    #                                  "num_files": num_files,
    #                                  "n": 2,
    #                                  "smoothing": "kneser-ney"},
    #                                 {"model": model,
    #                                  "num_files": num_files,
    #                                  "n": 3,
    #                                  "smoothing": "kneser-ney"}], save_data=True)

    # num_files = 50
    # scc.get_ngram_acc_perp_scatter([{"model": "STATISTICAL",
    #                                  "num_files": num_files,
    #                                  "n": 1,
    #                                  "smoothing": None},
    #                                 {"model": "STATISTICAL",
    #                                  "num_files": num_files,
    #                                  "n": 2,
    #                                  "smoothing": None},
    #                                 {"model": "STATISTICAL",
    #                                  "num_files": num_files,
    #                                  "n": 3,
    #                                  "smoothing": None},
    #                                 {"model": "STATISTICAL",
    #                                  "num_files": num_files,
    #                                  "n": 2,
    #                                  "smoothing": "absolute"},
    #                                 {"model": "STATISTICAL",
    #                                  "num_files": num_files,
    #                                  "n": 3,
    #                                  "smoothing": "absolute"},
    #                                 {"model": "STATISTICAL",
    #                                  "num_files": num_files,
    #                                  "n": 2,
    #                                  "smoothing": "Kneser-Ney"},
    #                                 {"model": "STATISTICAL",
    #                                  "num_files": num_files,
    #                                  "n": 3,
    #                                  "smoothing": "Kneser-Ney"}
    #                                 ], save_data=True)
