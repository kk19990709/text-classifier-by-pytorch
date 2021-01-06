'''
Author: your name
Date: 2021-01-06 17:32:56
LastEditTime: 2021-01-06 19:35:12
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /digital/model/Classifier.py
'''
import torch
import torch.nn as nn
from hyper import *

if torch.cuda.is_available() and opt.gpu >= 0:
    device = torch.device('cuda:' + str(opt.gpu))
else:
    device = torch.device('cpu')


class Classifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.input_size = INPUT_SIZE
        self.hidden_size = opt.hidden_size
        self.output_size = NUM_CLASSES
        self.batch_size = opt.batch_size
        self.num_layers = opt.num_layers
        self.embedding_layer = nn.Embedding(opt.max_nb_words + 2, self.input_size)
        self.lstm_layer = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc_layer1 = nn.Linear(self.hidden_size, 64)
        self.fc_layer2 = nn.Linear(64, 8)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        embed = self.embedding_layer(x)
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        hidden_states, _ = self.lstm_layer(embed, (h0, c0))
        mlp_input = hidden_states[:, -1, :]
        mlp_hidden = self.dropout(self.relu(self.fc_layer1(mlp_input)))
        mlp_hidden = self.fc_layer2(mlp_hidden)
        mlp_output = self.softmax(mlp_hidden)
        return mlp_output
