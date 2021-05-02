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

class LSTMNeuralLanguageModel:

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

        self.EPOCHS = 50
        self.N_HIDDEN = 512
        self.N_LAYERS = 4
        self.BATCH_SIZE = 10
        self.SEQ_LENGTH = 59
        self.CLIP = 5
        self.LEARNING_RATE = 0.001

        # Pre-processing hyper-parameters
        self.VAL_SPLIT = 0.4
        self.encoded = self.preprocess_data()
        val_idx = int(len(self.encoded) * (1 - self.VAL_SPLIT))
        self.train_data, self.val_data = self.encoded[:val_idx], self.encoded[val_idx:]


    def get_training_testing(self, train_data_set, split=0.8):
        filenames = os.listdir(train_data_set)
        n = len(filenames)
        print("There are {} files in the training directory: {}".format(n, train_data_set))
        np.random.shuffle(filenames)
        index = int(n * split)
        return filenames[:index], filenames[index:]

    def preprocess_data(self):
        chars = self.process_file(self.train_files)
        # corpus, chars, int2char, char2int = self.process_test_corpus()
        int2char = dict(enumerate(chars))
        char2int = {ch: ii for ii, ch in int2char.items()}
        return np.array([char2int[ch] for ch in chars])

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
        return tuple(set(corpus))

    def one_hot_encode(self, arr, n_labels):
        # Initialize the the encoded array
        one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
        # Fill the appropriate elements with ones
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
        # Finally reshape it to get back to the original array
        return one_hot.reshape((*arr.shape, n_labels))

    # get_batched returns batches of size (batch_size*seq_length).
    # Takes as input an array to make batches from, a batch size (the number of sequences
    # per batch) and seq_length which defines the number of encoded chars in a sequence
    def get_batches(self, arr, batch_size, seq_length):
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
        model = LSTMModel(self.encoded, self.N_HIDDEN, self.N_LAYERS)
        model.to(self.device)

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=self.LEARNING_RATE)

        if save_model:
            wandb.init(project="anle-cw")
            wandb.watch(model)

        loss_stats = {'train': [], 'val': []}

        n_chars = len(model.chars)
        print("Beginning training")
        for epoch in range(self.EPOCHS):
            model.train()
            # initialize hidden state
            train_hidden = model.init_hidden(self.BATCH_SIZE)

            for X_train, y_train in self.get_batches(self.train_data, self.BATCH_SIZE, self.SEQ_LENGTH):
                # One-hot encode our data and make them Torch tensors
                X_train = self.one_hot_encode(X_train, n_chars)
                X_train, y_train = torch.from_numpy(X_train).to(self.device), torch.from_numpy(y_train).to(self.device)

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                train_hidden = tuple([each.data for each in train_hidden])

                # zero accumulated gradients
                model.zero_grad()

                # get the output from the model
                y_train_pred, train_hidden = model(X_train, train_hidden)

                train_loss = loss_func(y_train_pred, y_train.view(self.BATCH_SIZE*self.SEQ_LENGTH))
                loss_stats['train'].append(train_loss)

                train_loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(model.parameters(), self.CLIP)
                optimizer.step()

            model.eval()
            val_hidden = model.init_hidden(batch_size)
            for X_val, y_val in self.get_batches(self.val_data, self.BATCH_SIZE, self.SEQ_LENGTH):
                # One-hot encode our data and make them Torch tensors
                X_val = self.one_hot_encode(X_val, n_chars)
                X_val, y_val = torch.from_numpy(X_val).to(self.device), torch.from_numpy(y_val).to(self.device)

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_hidden = tuple([each.data for each in val_hidden])

                y_val_pred, val_hidden = model(X_val, val_hidden)
                val_loss = criterion(y_val_pred, targets.view(self.BATCH_SIZE*self.SEQ_LENGTH))
                loss_stats['val'].append(val_loss)
            print(f"Epoch {(epoch+1)+0:02}: | Train Loss: {loss_stats['train'][-1]:.5f} | Val Loss: {loss_stats['val'][-1]:.5f}")
            if save_model:
                wandb.log({'Train Loss': loss_stats['train'][-1], 'Val Loss': loss_stats['val'][-1]})
        print("Finished Training")
        if save_model:
            save_path = f"NEURAL_MODELS/LSTM"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), f"{save_path}/{wandb.run.name}.pth")
        return model


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
        # you may need to use contiguous to reshape the output
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
    methodparams = {"model": "NEURAL",
                    "num_files": 1,
                    "test_model": False}

    NLM = LSTMNeuralLanguageModel(methodparams)
    NLM.train_model(save_model=False)


