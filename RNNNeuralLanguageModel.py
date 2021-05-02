import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from nltk import word_tokenize as tokenize
from nltk import ngrams
from collections import deque
from sklearn.model_selection import train_test_split
from collections import defaultdict
import wandb


class RNNNeuralLanguageModel:

    def __init__(self, methodparams):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"RUNNING ON: {self.device}")

        # Seeds
        np.random.seed(101)
        torch.manual_seed(101)

        self.train_data_set = "Holmes_data_set"
        self.training_files, self.testing_files = self.get_training_testing(self.train_data_set)

        self.train_files = self.training_files[:methodparams.get("num_files")]
        # self.test_files = self.testing_files[:methodparams.get("num_files")]

        # Pre-processing hyper-parameters
        self.int2char, self.char2int, self.corpus, self.X, self.y = self.preprocess_data()

        # MLP hyper-parameters
        self.INPUT_SIZE = len(self.char2int)
        self.OUTPUT_SIZE = len(self.char2int)
        self.BATCH_SIZE = len(self.corpus)
        self.HIDDEN_DIM = 100
        self.N_LAYERS = 5
        self.EPOCHS = 100
        self.LEARNING_RATE = 0.01


    def get_training_testing(self, train_data_set, split=0.8):
        filenames = os.listdir(train_data_set)
        n = len(filenames)
        print("There are {} files in the training directory: {}".format(n, train_data_set))
        np.random.shuffle(filenames)
        index = int(n * split)
        return filenames[:index], filenames[index:]

    def preprocess_data(self):
        corpus, chars, int2char, char2int = self.process_file(self.train_files)
        # corpus, chars, int2char, char2int = self.process_test_corpus()
        # corpus, max_len = self.naively_pad_sentences(corpus)
        corpus, max_len = self.better_pad_sentences(corpus)
        print(corpus[:10])
        input_seq, target_seq = self.split_data(corpus)
        # input_seq, target_seq = self.convert_char2int(corpus, input_, target_, char2int)
        for i in range(len(corpus)):
            input_seq[i] = [char2int[character] for character in input_seq[i]]
            target_seq[i] = [char2int[character] for character in target_seq[i]]
        return int2char, char2int, corpus, \
               torch.from_numpy(self.one_hot_encode(input_seq, len(char2int), max_len-1, len(corpus))), \
               torch.Tensor(target_seq)

    def process_test_corpus(self):
        corpus = ['hey how are you', 'good i am fine', 'have a nice day', 'good tidings we bring']
        # Join all the sentences together and extract the unique characters from the combined sentences
        chars = set(''.join(corpus))
        # Creating a dictionary that maps integers to the characters
        int2char = dict(enumerate(chars))
        # Creating another dictionary that maps characters to integers
        char2int = {char: ind for ind, char in int2char.items()}
        return corpus, chars, int2char, char2int

    def process_file(self, files):
        corpus = []
        for file in files:
            print("Processing {}".format(file))
            try:
                with open(os.path.join(self.train_data_set, file)) as in_stream:
                    for line in in_stream:
                        line = line.rstrip()
                        if len(line) > 0:
                            corpus.append(line)
            except UnicodeDecodeError:
                print("UnicodeDecodeError processing {}: ignoring file".format(file))
        chars = set(''.join(corpus))
        int2char = dict(enumerate(chars))
        char2int = {char: ind for ind, char in int2char.items()}
        return corpus, chars, int2char, char2int

    def naively_pad_sentences(self, corpus):
        max_len = len(max(corpus, key=len))
        for i in range(len(corpus)):
            while len(corpus[i]) < max_len:
                corpus[i] += ' '
        return corpus, max_len

    def better_pad_sentences(self, corpus):
        lengths = [len(i) for i in corpus]
        avg_len = round((float(sum(lengths)) / len(lengths)))
        for i in range(len(corpus)):
            while len(corpus[i]) < avg_len:
                corpus[i] += ' '
            corpus[i] = corpus[i][:avg_len]
        return corpus, avg_len


        # Creating lists that will hold our input and target sequences
    def split_data(self, corpus):
        input_seq, target_seq = [], []
        for i in range(len(corpus)):
            # Remove last character for input sequence
            input_seq.append(corpus[i][:-1])
            # Remove first character for target sequence
            target_seq.append(corpus[i][1:])
            print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))
        return input_seq, target_seq

    def convert_char2int(self, corpus, X, y, char2int):
        for i in range(len(corpus)):
            X[i] = [char2int[character] for character in X[i]]
            y[i] = [char2int[character] for character in y[i]]
        return X, y

    def one_hot_encode(self, sequence, dict_size, seq_len, batch_size):
        # Creating a multi-dimensional array of zeros with the desired output shape
        features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

        # Replacing the 0 at the relevant character index with a 1 to represent that character
        for i in range(batch_size):
            for u in range(seq_len):
                features[i, u, sequence[i][u]] = 1
        return features

    def train_model(self, save_model=False):
        model = RNNModel(self.INPUT_SIZE, self.OUTPUT_SIZE, self.HIDDEN_DIM, self.N_LAYERS)
        model.to(self.device)

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=self.LEARNING_RATE)

        if save_model:
            wandb.init(project="anle-cw")
            wandb.watch(model)

        loss_stats = {'train': []}

        print("Beginning training")
        for epoch in range(self.EPOCHS):
            model.train()
            # train_epoch_loss, train_epoch_acc = 0, 0
            # for context, target in self.n_gram.items():
                # optimizer.zero_grad()
                # if type(context) is tuple or list:
                #     context_idxs = torch.tensor([self.word_to_idx[w] for w in context], dtype=torch.long).to(self.device)
                # else:
                #     context_idxs = torch.tensor(self.word_to_idx[context], dtype=torch.long).to(self.device)
                # log_probs = model(context_idxs)
                #
                # train_loss = loss_func(log_probs, torch.tensor([self.word_to_idx[target]], dtype=torch.long).to(self.device))
                #
                # train_loss.backward()
                # optimizer.step()
                # train_epoch_loss += train_loss.item()

            X_train, y_train = self.X.to(self.device), self.y.to(self.device)
            optimizer.zero_grad()

            y_train_pred, hidden = model(X_train)

            train_loss = loss_func(y_train_pred, y_train.view(-1).long())
            loss_stats['train'].append(train_loss)

            train_loss.backward()
            optimizer.step()


            print(f"Epoch {(epoch+1)+0:02}: | Train Loss: {loss_stats['train'][-1]:.5f}")
            if save_model:
                wandb.log({'Train Loss': loss_stats['train'][-1]})
        print("Finished Training")
        if save_model:
            save_path = f"NEURAL_MODELS"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), f"{save_path}/{wandb.run.name}.pth")
        return model

    def load_model(self, model_path):
        model = RNNModel(self.INPUT_SIZE, self.OUTPUT_SIZE, self.HIDDEN_DIM, self.N_LAYERS)
        model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        print("MODEL LOADED")
        return model

    # This function takes in character as argument and returns the next character prediction and hidden state
    def predict(self, model, character):
        # One-hot encoding our input to fit into the model
        character = np.array([[self.char2int[c] for c in character]])
        character = torch.from_numpy(self.one_hot_encode(character, len(self.char2int), character.shape[1], 1))
        character = character.to(self.device)

        out, hidden = model(character)

        prob = nn.functional.softmax(out[-1], dim=0).data
        # Taking the class with the highest probability score from the output
        char_ind = torch.max(prob, dim=0)[1].item()

        return self.int2char[char_ind], hidden

    # This function takes the desired output length and input characters as arguments, returning the produced sentence
    def sample(self, model, out_len, start='hey'):
        model.eval()  # eval mode
        start = start.lower()
        # First off, run through the starting characters
        chars = [ch for ch in start]
        size = out_len - len(chars)
        # Now pass in the previous characters and get a new one
        for ii in range(size):
            char, h = self.predict(model, chars)
            chars.append(char)
        return ''.join(chars)

class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        nn.Module.__init__(self)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden


if __name__ == "__main__":
    methodparams = {"model": "NEURAL",
                    "num_files": 1,
                    "test_model": False}

    NLM = RNNNeuralLanguageModel(methodparams)
    # NLM.train_model(save_model=True)
    # model = NLM.load_model("NEURAL_MODELS/lyric-durian-7.pth")
    print(NLM.sample(NLM.train_model(save_model=False), 15, "good"))
