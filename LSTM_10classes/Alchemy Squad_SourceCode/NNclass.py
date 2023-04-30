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


# Create Convolutional Network
# Originally, when input_size = 100 (1x10x10 image), 
# conv1 -> 8x10x10, because out_channels = 8, i.e. 8 output channels. 
# relu does not change x shape!!
# pool -> 8x5x5, performs a 2x2 max pooling
# relu(conv2) -> 16x5x5, out_channels = 16
# pool -> 16x2x2
# Therefore, fc1 use a Linear(16*7*7). If input_size changes, this should change.

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
        self.fc1 = nn.Linear(hidden_size, 10)
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, x):  # , lengths):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x, _ = self.lstm(x)
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
