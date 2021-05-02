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

import wandb

class ANN_Language_Model:

    def __init__(self, num_training_files, lm_flag):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"RUNNING ON: {self.device}")

        # Seeds
        np.random.seed(101)
        torch.manual_seed(101)

        train_data_set = "Holmes_data_set"
        self.training_files, self.held_out_files = self.get_training_testing(train_data_set)

        self.train_data_set = train_data_set
        self.files = self.training_files[:num_training_files]

        self.lm_flag = lm_flag
        if self.lm_flag == "N_GRAM_ANN":
            self.n = int(input("PLEASE SPECIFY ANN N_GRAM SIZE: "))
            self.vocab = set()
            self.n_gram = {}

            test_sentence = """When forty winters shall besiege thy brow,
            And dig deep trenches in thy beauty's field,
            Thy youth's proud livery so gazed on now,
            Will be a totter'd weed of small worth held:
            Then being asked, where all thy beauty lies,
            Where all the treasure of thy lusty days;
            To say, within thine own deep sunken eyes,
            Were an all-eating shame, and thriftless praise.
            How much more praise deserv'd thy beauty's use,
            If thou couldst answer 'This fair child of mine
            Shall sum my count, and make my old excuse,'
            Proving his beauty by succession thine!
            This were to be new made when thou art old,
            And see thy blood warm when thou feel'st it cold."""
            self._process_line(test_sentence)

            # self._process_files()

            self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}

            print(self.n_gram)

            # MLP hyper-parameters
            self.VOCAB_SIZE = len(self.vocab)
            self.EMBEDDING_DIM = 300
            self.CONTEXT_SIZE = self.n - 1

            self.VAL_SIZE = 0.4
            self.EPOCHS = 100
            self.LEARNING_RATE = 0.001

        else:
            print(f"{self.lm_flag} NOT IMPLEMENTED")


    def get_training_testing(self, train_data_set, split=0.5):
        filenames = os.listdir(train_data_set)
        n = len(filenames)
        print("There are {} files in the training directory: {}".format(n, train_data_set))
        np.random.shuffle(filenames)
        index = int(n * split)
        return filenames[:index], filenames[index:]

    def _process_line(self, line):
        tokens = tokenize(line)

        # Adding sentence to vocab
        [self.vocab.add(token) for token in tokens]

        # Create n_gram
        for gram in ngrams(tokens, self.n):
            if self.n < 3:
                self.n_gram[list(gram)[0]] = gram[-1]
            else:
                self.n_gram[tuple(list(gram)[:-1])] = gram[-1]

    def _process_files(self):
        for file in self.files:
            print("Processing {}".format(file))
            try:
                with open(os.path.join(self.train_data_set, file)) as in_stream:
                    for line in in_stream:
                        line = line.rstrip()
                        if len(line) > 0:
                            self._process_line(line)
            except UnicodeDecodeError:
                print("UnicodeDecodeError processing {}: ignoring file".format(file))

    def multi_acc(self, y_pred, y_test):
        _, y_pred_tags = torch.max(torch.log_softmax(y_pred, dim=1), dim=1)
        correct_pred = (y_pred_tags == y_test).float()
        return torch.round(correct_pred.sum() / len(correct_pred)*100)


    def train_model(self, save_path="ANN_MODELS", save_model=True):
        model = MLPModel(self.VOCAB_SIZE, self.EMBEDDING_DIM, self.CONTEXT_SIZE)
        model.to(self.device)

        loss_func = nn.NLLLoss()
        optimizer = optim.SGD(params=model.parameters(), lr=self.LEARNING_RATE)

        if save_model:
            wandb.init(project=f"ANLE-CW-{self.lm_flag}")
            wandb.watch(model)

        loss_stats = {'train': []}

        print("Beginning training")
        for epoch in range(self.EPOCHS):
            model.train()
            train_epoch_loss, train_epoch_acc = 0, 0
            for context, target in self.n_gram.items():
                optimizer.zero_grad()
                if type(context) is tuple:
                    context_idxs = torch.tensor([self.word_to_idx[w] for w in context], dtype=torch.long).to(self.device)
                else:
                    context_idxs = torch.tensor(self.word_to_idx[context], dtype=torch.long).to(self.device)
                log_probs = model(context_idxs)

                train_loss = loss_func(log_probs, torch.tensor([self.word_to_idx[target]], dtype=torch.long).to(self.device))

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
            loss_stats['train'].append(train_epoch_loss / len(self.n_gram.items()))
            print(f"Epoch {(epoch+1)+0:02}: | Train Loss: {loss_stats['train'][-1]:.5f}")
            if save_model:
                wandb.log({'Train Loss': loss_stats['train'][-1]})
        print("Finished Training")
        if save_model:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), f"{save_path}/{wandb.run.name}.pth")

    def get_prob(self, token, context=None, methodparams=None):
        model_path = input("PLEASE INPUT A PATH TO MODEL: ")
        model = self.model_class
        model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        print("MODEL LOADED")
        model.eval()
        if type(context) is tuple:
            context_idxs = torch.tensor([self.word_to_idx[w] for w in context], dtype=torch.long).to(self.device)
        else:
            context_idxs = torch.tensor(self.word_to_idx[context], dtype=torch.long).to(self.device)
        print(model(context_idxs))



    def load_model(self, model_path):
        print("Loading model...")
        model = self.model_class
        model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        self.model = model
        print("Calculating training route view latent spaces...")
        self.route_view_layton_spaces = []
        for filename in probar(self.route_filenames):
            self.model(self.transform(Image.fromarray(self.preprocess(cv2.imread(self.route_path + filename)))).float().to(self.device).view(1, self.INPUT_SIZE))
            self.route_view_layton_spaces.append(self.model.get_latent_space())

    def test_model(self):
        y_pred, y_ground_truth = [], []
        with torch.no_grad():
            for X_test_batch, y_test_batch in self.testloader:
                X_test_batch, y_test_batch = X_test_batch.to(self.device), y_test_batch.to(self.device)

                y_test_pred = self.model(X_test_batch)
                # print(y_test_pred)
                _, y_pred_tag = torch.max(y_test_pred, dim=1)

                y_pred.append(y_pred_tag.cpu().numpy())
                y_ground_truth.append(y_test_batch.cpu().numpy())
        print(classification_report(y_ground_truth, y_pred, zero_division=0))


class MLPModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        nn.Module.__init__(self)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

if __name__ == "__main__":
    num_training_files = 1
    ann_flag = "N_GRAM_ANN"
    save = True

    ann_lm = ANN_Language_Model(num_training_files, ann_flag)
    # ann_lm.train_model(save_path=f"ANN_MODELS/{ann_flag}", save_model=save)