import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from nltk import word_tokenize as tokenize
import nltk
from nltk import ngrams
from collections import deque
from sklearn.model_selection import train_test_split
from collections import defaultdict
import wandb
import csv
import pandas as pd
from nltk.corpus import wordnet as wn, wordnet_ic as wn_ic, lin_thesaurus as lin
import re
from pyprobar import probar


class NeuralLanguageModel:

    def __init__(self, params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"RUNNING ON: {self.device}")

        # Seeds
        # np.random.seed(101)
        # torch.manual_seed(101)

        self.params = params
        self.model_flag = self.params.get("model")

        self.EPOCHS = 50
        self.N_HIDDEN = 300
        self.N_LAYERS = 3
        self.BATCH_SIZE = 64
        self.SEQ_LENGTH = 160
        self.CLIP = 5
        self.LEARNING_RATE = 0.01

        self.train_data_set = "Holmes_data_set"
        self.training_files, self.testing_files = self._get_training_testing(self.train_data_set)

        num_files = self.params.get("num_files")
        if num_files is None:
            print("TESTING MODEL: NO TRAIN FILES LOADED")
        else:
            self.train_files = self.training_files[:]
            self.VAL_SPLIT = 0.4
            self.chars, self.encoded = self._preprocess_data()
            val_idx = int(len(self.encoded) * (1 - self.VAL_SPLIT))
            self.train_data, self.val_data = self.encoded[:val_idx], self.encoded[val_idx:]

    def _get_training_testing(self, train_data_set, split=0.8):
        filenames = os.listdir(train_data_set)
        n = len(filenames)
        print("There are {} files in the training directory: {}".format(n, train_data_set))
        np.random.shuffle(filenames)
        index = int(n * split)
        return filenames[:index], filenames[index:]

    def _preprocess_data(self):
        text = self._process_file(self.train_files)
        text = self._clean_text(text)
        chars = tuple(set(text))
        int2char = dict(enumerate(chars))
        char2int = {ch: ii for ii, ch in int2char.items()}
        return chars, np.array([char2int[ch] for ch in text])

    def _clean_text(self, text):
        # tokenizer = nltk.RegexpTokenizer(r"\w+")
        text = [word for word in nltk.word_tokenize(text) if word.isalnum()]
        return ' '.join(text)

    def _process_file(self, files):
        corpus = []
        print(f"PROCESSING {self.params.get('num_files')} FILES")
        for file in probar(files):
            try:
                with open(os.path.join(self.train_data_set, file)) as in_stream:
                    for line in in_stream:
                        line = line.rstrip()
                        if len(line) > 0:
                            corpus.append(line)
            except UnicodeDecodeError:
                print("UnicodeDecodeError processing {}: ignoring file".format(file))
        return ' '.join(corpus)

    def _one_hot_encode(self, arr, n_labels):
        # Initialize the the encoded array
        one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
        # Fill the appropriate elements with ones
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
        # Finally reshape it to get back to the original array
        return one_hot.reshape((*arr.shape, n_labels))

    # get_batched returns batches of size (batch_size*seq_length).
    # Takes as input an array to make batches from, a batch size (the number of sequences
    # per batch) and seq_length which defines the number of encoded chars in a sequence
    def _get_batches(self, arr, batch_size, seq_length):
        batch_size_total = batch_size * seq_length
        # total number of batches we can make, // integer division, round down
        n_batches = len(arr) // batch_size_total
        # Keep only enough characters to make full batches
        arr = arr[:n_batches * batch_size_total]
        # Reshape into batch_size rows, n. of first row is the batch size, the other lenght is inferred
        arr = arr.reshape((batch_size, -1))
        # iterate through the array, one sequence at a time
        for n in range(0, arr.shape[1], seq_length):
            # The features
            x = arr[:, n:n + seq_length]
            # The targets, shifted by one
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_length]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
            yield x, y

    def train_model(self, save_model=False):
        if self.model_flag == "NEURAL-RNN":
            model = RNNModel(tokens=self.chars, n_hidden=self.N_HIDDEN, n_layers=self.N_LAYERS)
        elif self.model_flag == "NEURAL-LSTM":
            model = LSTMModel(tokens=self.chars, n_hidden=self.N_HIDDEN, n_layers=self.N_LAYERS)
        else:
            print("MODEL TYPE NOT UNDERSTOOD")
            return None
        model.to(self.device)
        print(f"MODEL ARCHITECTURE:\n{model}")

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=self.LEARNING_RATE)

        save_path = f"NEURAL_MODELS/{self.model_flag}"
        if save_model:
            wandb.init(project="anle-cw")
            wandb.watch(model)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

        loss_stats = {'train': [], 'val': []}

        n_chars = len(model.chars)
        print("Beginning training")
        for epoch in range(self.EPOCHS):

            model.train()
            train_hidden = model.init_hidden(self.BATCH_SIZE)
            for X_train, y_train in self._get_batches(self.train_data, self.BATCH_SIZE, self.SEQ_LENGTH):
                X_train = self._one_hot_encode(X_train, n_chars)
                X_train, y_train = torch.from_numpy(X_train).to(self.device), torch.from_numpy(y_train).to(self.device)

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                train_hidden = tuple([each.data for each in train_hidden])

                model.zero_grad()

                y_train_pred, train_hidden = model(X_train, train_hidden)

                train_loss = loss_func(y_train_pred, y_train.view(self.BATCH_SIZE * self.SEQ_LENGTH).long())
                loss_stats['train'].append(train_loss.item())

                train_loss.backward()
                # clip_grad_norm helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(model.parameters(), self.CLIP)
                optimizer.step()

                if save_model and epoch % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss}, f"{save_path}/{wandb.run.name}_epoch{epoch}.pth")

            model.eval()
            val_hidden = model.init_hidden(self.BATCH_SIZE)
            for X_val, y_val in self._get_batches(self.val_data, self.BATCH_SIZE, self.SEQ_LENGTH):
                X_val = self._one_hot_encode(X_val, n_chars)
                X_val, y_val = torch.from_numpy(X_val).to(self.device), torch.from_numpy(y_val).to(self.device)

                val_hidden = tuple([each.data for each in val_hidden])

                y_val_pred, val_hidden = model(X_val, val_hidden)
                val_loss = loss_func(y_val_pred, y_val.view(self.BATCH_SIZE * self.SEQ_LENGTH).long())
                loss_stats['val'].append(val_loss.item())
            print(
                f"Epoch {(epoch + 1) + 0:02}: | Train Loss: {loss_stats['train'][-1]:.5f} | Val Loss: {loss_stats['val'][-1]:.5f}")
            if save_model:
                wandb.log({'Train Loss': loss_stats['train'][-1], 'Val Loss': loss_stats['val'][-1]})
        print("Finished Training")
        if save_model:
            # torch.save(model, f"{save_path}/{wandb.run.name}.pth")
            torch.save({
                'epoch': self.EPOCHS,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_stats['train'][-1]}, f"{save_path}/{wandb.run.name}_epoch{self.EPOCHS}.pth")
            field_names = ["TOKENS", "N_HIDDEN", "N_LAYERS"]
            model_data = [{"TOKENS": model.chars, "N_HIDDEN": self.N_HIDDEN, "N_LAYERS": self.N_LAYERS}]
            with open(f"{save_path}/{wandb.run.name}_data.csv", 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writeheader()
                writer.writerows(model_data)
        return model

    def load_model(self, model_path, model_data_path):
        model_data = pd.read_csv(model_data_path)
        tokens = re.findall(r"'(.*?)'", model_data["TOKENS"][0])
        if self.model_flag == "NEURAL-RNN":
            model = RNNModel(tokens, int(model_data["N_HIDDEN"][0]), int(model_data["N_LAYERS"][0]))
        else:
            model = LSTMModel(tokens, int(model_data["N_HIDDEN"][0]), int(model_data["N_LAYERS"][0]))
        model.to(self.device)

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        print("MODEL LOADED")
        return model

    # get_pred_chat takes a model and a character as argument
    # returns the next character prediction and hidden state
    def get_pred_char(self, model, character, hidden=None, top_k=None):
        character = np.array([[model.char2int[character]]])
        character = self._one_hot_encode(character, len(model.chars))
        character = torch.from_numpy(character).to(self.device)

        # detach hidden state from history
        hidden = tuple([each.data for each in hidden])
        out, hidden = model(character, hidden)

        # softmax used to get probabilities of the likely next character given out
        prob = F.softmax(out, dim=1).data.to(self.device)
        if self.device == torch.device('cuda'):
            prob = prob.cpu()  # move to cpu

        if top_k is None:
            top_ch = np.arange(len(model.chars))
        else:
            prob, top_ch = prob.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        prob = prob.numpy().squeeze()
        char_ind = np.random.choice(top_ch, p=prob / prob.sum())

        return model.int2char[char_ind], hidden

    # get_pred_word takes the model, a prime sentence and top_k (defining the scope of characters considered probable)
    # returns the predicted word followed by the full sentence
    def get_pred_word(self, model, prime='', top_k=None):
        model.eval()
        chars = [ch for ch in prime]
        hidden = model.init_hidden(1)
        prime_char = ""
        for ch in prime:
            prime_char, hidden = self.get_pred_char(model, ch, hidden, top_k=top_k)
        chars.append(prime_char)

        cont = True
        while cont:
            pred_char, hidden = self.get_pred_char(model, chars[-1], hidden, top_k=top_k)
            if len(tokenize(pred_char)) == 0:
                cont = False
                break
            chars.append(tokenize(pred_char)[0])
        # print(f"SENTENCE: {''.join(chars)}\nWORD: {tokenize(''.join(chars))[-1]}")
        return tokenize(''.join(chars))[-1], ''.join(chars)

    def get_prob(self, X, y, measure='path'):
        X_synsets = wn.synsets(str(X))
        # print(X_synsets)
        if len(X_synsets) == 0:
            return 0
        y_synsets = wn.synsets(str(y))
        if measure == 'lch':
            similarities = [X_.lch_similarity(y_) for X_ in X_synsets for y_ in y_synsets]
        elif measure == 'wup':
            print("CALLED")
            similarities = [X_.wup_similarity(y_) for X_ in X_synsets for y_ in y_synsets]
        elif measure == 'res':
            similarities = [X_.res_similarity(y_, wn_ic.ic('ic-brown.dat')) for X_ in X_synsets for y_ in y_synsets]
        elif measure == 'jcn':
            similarities = [X_.jcn_similarity(y_, wn_ic.ic('ic-brown.dat')) for X_ in X_synsets for y_ in y_synsets]
        elif measure == 'lin':
            similarities = [X_.lin_similarity(y_, wn_ic.ic('ic-brown.dat')) for X_ in X_synsets for y_ in y_synsets]
        else:
            # Else, calculate path similarity
            similarities = [X_.path_similarity(y_) for X_ in X_synsets for y_ in y_synsets]
        similarities = [i for i in similarities if i]
        return max(similarities)


class LSTMModel(nn.Module):

    def __init__(self, tokens, n_hidden=612, n_layers=4, drop_prob=0.5, lr=0.001):
        nn.Module.__init__(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Hyper-parameters
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # Layers
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    # Forward pass takes inputs (x) and the hidden state (hidden)
    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)

        # Stack up LSTM outputs using view
        out = out.contiguous().view(-1, self.n_hidden)

        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden

    # Initiliases the hidden state with tensors of zeros (size: n_layers*batch_size*n_hidden)
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device))
        return hidden


class RNNModel(nn.Module):

    def __init__(self, tokens, n_hidden=612, n_layers=4, drop_prob=0.5, lr=0.001):
        nn.Module.__init__(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Hyper-parameters
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # Layers
        self.rnn = nn.RNN(len(self.chars), n_hidden, n_layers,
                          dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    # Forward pass takes inputs (x) and the hidden state (hidden)
    def forward(self, x, hidden):
        r_output, hidden = self.rnn(x, hidden)
        out = self.dropout(r_output)

        # Stack up RNN outputs using view
        out = out.contiguous().view(-1, self.n_hidden)

        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden

    # Initiliases the hidden state with tensors of zeros (size: n_layers*batch_size*n_hidden)
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device))
        return hidden


if __name__ == "__main__":
    params = {"model": "NEURAL-RNN",
              "num_files": 10}

    NLM = NeuralLanguageModel(params)

    NLM.train_model(save_model=True)

    # model = NLM.load_model(model_path="NEURAL_MODELS/LSTM/jolly-bush-44_epoch50.pth",
    #                        model_data_path="NEURAL_MODELS/LSTM/jolly-bush-44_data.csv")
    # pred_word, sentence = NLM.get_pred_word(model, prime='I have it from the same source that you are both an orphan and a bachelor and are ', top_k=5)
    # print(NLM.get_prob("the", "discipline", measure='path'))
