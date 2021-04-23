from SentenceCompletionChallenge import SentenceCompletionChallenge
import nltk

class Trigram(SentenceCompletionChallenge):
    def __init__(self, data_set_dir):
        SentenceCompletionChallenge.__init__(self, data_set_dir)

if __name__ == '__main__':
    data_set_dir = ""
    trigram = Trigram(data_set_dir)

    # print(trigram.q_df["left_context"].head())
    # print(trigram.questions[0].answer)