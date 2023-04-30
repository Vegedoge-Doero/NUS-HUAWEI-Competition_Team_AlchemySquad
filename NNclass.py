import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import math
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)  # matrix 784x50
        self.fc2 = nn.Linear(50, num_classes)  # matrix 50x10

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*2*2, num_classes)
        #self.fc1 = nn.Linear(16*7*7, num_classes)
        # Originally, when input_size = 784 (1x28x28 image), 
        # conv1 -> 8x28x28, because out_channels = 8, i.e. 8 output channels. relu does not change x shape.
        # pool -> 8x14x14, performs a 2x2 max pooling
        # relu(conv2) -> 16x14x14, out_channels = 16
        # pool -> 16x7x7
        # Therefore, fc1 use a Linear(16*7*7). If input_size changes, this should change.

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# Create Recurrent Network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length, device):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float).to(self.device)
        x, _ = self.rnn(x, h0)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


# Create GRU Network
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length, device):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        x, _ = self.gru(x, h0)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


# Create LSTM Network
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):  # , lengths):
        x, _ = self.lstm(x)
        x = F.relu(self.fc1(x[:, -1, :]))
        x = self.fc2(x)
        return x

# To add an Attention layer, we are supposed to DEFINE the attention by ourselves!
# GPT suggest "self-attention" (scaled dot-product attention)
# i.e. computing an attention weight for each time step in the input sequence, and 
# then taking a weighted average of the LSTM hidden states using these weights.

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        # query, key, value = project the input tensor x to three different spaces of dimension 'hidden_size'.

    def forward(self, x):
        # batch_size and seq_len are inferred from the input tensor x
        # shape of x: (batch_size, seq_len, hidden_size)

        Q = self.query(x)  # (batch_size, seq_len, hidden_size)
        K = self.key(x)  # (batch_size, seq_len, hidden_size)
        # compute the pairwise similarities between each time step in the input sequence, based on how similar they are to the current query vector.

        V = self.value(x)  # (batch_size, seq_len, hidden_size)
        # combined with the attention weights to obtain a context vector.
        # represents a weighted sum of the value vectors, where the weights are given by the attention weights.

        weights = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_size)  # (batch_size, seq_len, seq_len)
        # torch.matmul -> matrix multiplication
        # K.transpose(1, 2) -> transpose the last 2 dimensions of K, now (batch_size, hidden_size, seq_len)
        # divisor -> computes the dot product between each query vector and each key vector, resulting in a tensor of shape (batch_size, seq_len, seq_len)
        # math.sqrt(self.hidden_size) -> scaling factor

        weights = F.softmax(weights, dim=2)  # (batch_size, seq_len, seq_len)
        # normalisation using softmax

        output = torch.matmul(weights, V)  # (batch_size, seq_len, hidden_size)
        return output

class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTM1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):  # , lengths):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        x, _ = self.lstm(x, (h0, c0))
        x = self.attention(x)
        x = F.relu(self.fc1(x[:, -1, :]))
        x = self.fc2(x)
        return x


# The one WITHOUT Conv layer
class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        x, _ = self.lstm(x, (h0, c0))
        x = self.fc(x[:, -1, :])
        return x

# The one WITH Conv layer
class BLSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(BLSTM1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.conv = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=10, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x, (h0, c0))
        x = self.fc(x[:, -1, :])
        return x


# Save/Load Network
def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('=> Saving Checkpoint!')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print('=> Loading Checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Modifying Classifier
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
