import random
import numpy as np
import json
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
import sys
from testRandom import RandomTensor
from testStocks import StocksTensor, pickData
from NNclass import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set Device
load_model = False
save_model = True
save_step = 2
loadpoint = 2
abortpoint = 10000
size0 = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 100
num_classes = 20
learning_rate = 0.001
batch_size = 64
num_epochs = 200
in_channels = 1  # for CNN
sequence_length = 10  # for RNN
input_size_RNN = 50  # for RNN
num_layers = 2  # for RNN
# hidden_size = 256  # for RNN
# better use hidden_size = 100 - datasize match
hidden_size = 100
size = 10000  # for random dataset

# Initialize

# For different networks, you need to use different input sizes!
# NN: myTrainModel.py, input_size = 100 -- since each data input length = 100 (64%)
# CNN: NNclass.py, self.fc1 = nn.Linear(16*2*2) (68%)
# RNN: myTrainModel.py, input_size_RNN = 100 (70.3%)
# GRU: myTrainModel.py, input_size_RNN = 100 (61.5%)
# LSTM: 61.4%
# BLSTM: around 75%
# VGG16: around 75%

# transformer = another version. CNN is for picture processing
# only fc = NN; conv and fc = CNN
# actually the only difference is layer, but the layer can be re-organised

# We are actually predicting the LAST data of the 100 data point
# and then compare with the VWAP (average price) of the day
# if the last data is 28, VWAP of the day is 42. Then currently is at its low, should buy
# if there are buying actions, it will happen at the end of the 100 ticks

useNN = 'LSTM'
if useNN == 'NN':
    model = NN(input_size=input_size, num_classes=num_classes).to(device=device)
elif useNN == 'CNN':
    model = CNN(in_channels=in_channels, num_classes=num_classes).to(device=device)
elif useNN == 'RNN':
    model = RNN(input_size=input_size_RNN, hidden_size=hidden_size, num_layers=num_layers,
                num_classes=num_classes, sequence_length=sequence_length, device=device).to(device)
elif useNN == 'GRU':
    model = GRU(input_size=input_size_RNN, hidden_size=hidden_size, num_layers=num_layers,
                num_classes=num_classes, sequence_length=sequence_length, device=device).to(device)
elif useNN == 'LSTM':
    model = LSTM1(input_size=input_size_RNN, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes,
                 device=device).to(device)
elif useNN == 'BLSTM':
    model = BLSTM(input_size=input_size_RNN, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes,
                  device=device).to(device)
elif useNN == 'BLSTM1':
    model = BLSTM1(input_size=input_size_RNN, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes,
                  device=device).to(device)
elif useNN == 'VGG16':
    model = torchvision.models.vgg16(pretrained=False)
    model.features[0] = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.features[9] = nn.MaxPool2d(kernel_size=1, stride=2, padding=0, dilation=1, ceil_mode=False)
    model.features[23] = nn.MaxPool2d(kernel_size=1, stride=2, padding=0, dilation=1, ceil_mode=False)
    model.features[30] = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    frozen_layer = 0
    for i, param in enumerate(model.parameters()):
        if i < frozen_layer:
            param.requires_grad = False
        else:
            param.requires_grad = True
    model.avgpool = Identity()
    model.classifier = nn.Sequential(nn.Linear(64, 32), nn.Dropout(p=0.5), nn.Linear(32, 10))
    model.to(device)
    print(model)
else:
    print('The network typed in is not valid!')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if load_model:
    load_checkpoint(torch.load(f'./SavedNN_multiCSV/{useNN}_checkpoint{loadpoint}.pth.tar'), model, optimizer)

# Datasets MNIST
loadStocks = True
if loadStocks:
    with open("../dataset_json_priceList/priceListtickdata_20220805.csv.json", "r") as fp:
        priceList = json.load(fp)
    with open("../dataset_json_msList/msListtickdata_20220805.csv.json", "r") as fp:
        msList = json.load(fp)
    with open("../dataset_json_VWAP/VWAPtickdata_20220805.csv.json", "r") as fp:
        VWAP = json.load(fp)
    with open("../dataset_json_stock_name/stock_nametickdata_20220805.csv.json", "r") as fp:
        stock_name = json.load(fp)
    dataset = StocksTensor(size=size, transform=transforms.ToTensor(), train=False, window=size0**2,
                           priceList=priceList, msList=msList, VWAP=VWAP, stock_name=stock_name)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [8000, len(dataset) - 8000])
    # the random split, choose a random BATCH
    # e.g. first time, I choose 64 data (each 100 points) from 1 stocks
    # second time, choose 64 data (1 batch) from another stocks
    # before random_split, it is jsut half-random, for each batch of 64 data, it belongs to the SAME stocks
    # after it, for the 64 piece of data each batch, it contains data from different stocks
    # batch = for each epoch, you divide your data into 64 groups, it gives 64 feedback loops
    # So now we have di(r) = di(r1) - di(r2), i is random
    train_dataset.train = True
    test_dataset.train = False
    print("Successfully loaded the data!")
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# 1x100 -> 10x10 -> 64x1x10x10
# During the TESTING, you DO NOT KNOW the VWAP (the average price of the day)!!

# During training, you know VWAP. If you buy at a price lower than VWAP,
# then we are selling at a price higher than VWAP, then we are earning! Regardless of what the price THEN is.

# We can even short-buy and short-sell, i.e. even if you don't have 50 shares of stock A, 
# you can still sell 50 shares of A at a certain point

# BLSTM:
# originally, data_input is (64x1x100)
# Each particular data point is 64x1x1
# Then if we use 5 line of data, each data point becomes 64x5x1
# This makes data_input becomes 64x5x100!!
# This makes in_channel = 5, rather than = 1
view_dataset = False
if view_dataset:
    for (images, targets) in train_loader:  # data_size = [64, 1, 32, 32]
        print(images.shape)
        for i in range(1, 10):
            pos = int('33' + str(i))
            plt.subplot(pos)
            plt.imshow(images[i, 0])
            plt.title(int(targets[i]))
        plt.show()

# Training Network
loss_list = []
epochs = range(1+loadpoint, 1+num_epochs) if load_model else range(1, 1+num_epochs)
loadMultipleCSV = True
numCSV = 99
dir_list_msList = os.listdir("../dataset_json_msList")
dir_list_priceList = os.listdir("../dataset_json_priceList")
dir_list_VWAP = os.listdir("../dataset_json_VWAP")
dir_list_stock_name = os.listdir("../dataset_json_stock_name")
for epoch in epochs:

    if loadMultipleCSV:
        r = random.randint(0, numCSV)
        print(dir_list_stock_name[r])
        with open("../dataset_json_priceList/"+dir_list_priceList[r], "r") as fp:
            priceList = json.load(fp)
        with open("../dataset_json_msList/"+dir_list_msList[r], "r") as fp:
            msList = json.load(fp)
        with open("../dataset_json_VWAP/"+dir_list_VWAP[r], "r") as fp:
            VWAP = json.load(fp)
        with open("../dataset_json_stock_name/"+dir_list_stock_name[r], "r") as fp:
            stock_name = json.load(fp)
        dataset = StocksTensor(size=size, transform=transforms.ToTensor(), train=False, window=size0 ** 2,
                               priceList=priceList, msList=msList, VWAP=VWAP, stock_name=stock_name)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [8000, len(dataset) - 8000])
        train_dataset.train = True
        test_dataset.train = False
        print("Successfully loaded the data!")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    running_loss = 0.0
    losses = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        targets = targets.squeeze(1)
        # print(targets)
        data = data.to(device=device)
        targets = targets.to(device=device)
        if useNN == 'NN':
            data = data.reshape(data.shape[0], -1)
        if useNN in ['RNN', 'GRU', 'LSTM', 'BLSTM']:
            data = data.squeeze(1)
        # froward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())
        running_loss += loss.item()
        # print(scores, loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss = running_loss / len(train_loader)
    print(f"{epoch} epoch training finished! ({epoch} / {num_epochs})")
    print('Epoch loss {:4f}'.format(epoch_loss))
    loss_list.append(epoch_loss)
    if epoch % save_step == 0 and save_model:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint, filename=f'./SavedNN_multiCSV/CSV{numCSV}_{useNN}_class{num_classes}_checkpoint{epoch}.pth.tar')
    if epoch == abortpoint:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint, filename=f'./SavedNN_multiCSV/CSV{numCSV}_{useNN}_class{num_classes}_checkpoint{epoch}.pth.tar')
        sys.exit("The training process is aborted and saved!")


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    if loader.dataset.train:
        print("Check accuracy on training data!")
    else:
        print("Check accuracy on test data!")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            if useNN == 'NN':
                x = x.reshape(x.shape[0], -1)
            if useNN in ['RNN', 'GRU', 'LSTM', 'BLSTM']:
                x = x.squeeze(1)
                # When dealing with data, originally reshape 1x100 into 10x10
                # but now squeeze back into 1x100
                # so input is still 1x100
            scores = model(x)
            _, predictions = scores.max(1)
            # num_correct += (predictions == y.t()).sum()
            y = y.t()
            indicesB = y < 9
            indicesS = y > 10
            indicesN = torch.logical_and(y >= 9, y <= 10)
            num_correct += (predictions[indicesB[0]] <= 8).sum()  # Earn for >=, not necessarily =
            num_correct += (predictions[indicesS[0]] >= 11).sum()
            num_correct += (torch.logical_and(predictions[indicesN[0]] >= 9, predictions[indicesN[0]] <= 10)).sum()
            # Here prediction scores have N row, 1 col; y has 1 row, N col
            # So directly compare will make it NxN! Which is incorrect
            # y.t() = y.transpose()! So we can make the correct comparison
            num_samples += predictions.size(0)  #predictions.size() = tensor([64])
        print(f'Got {num_correct} / {num_samples} with accuracy '
              f'{float(num_correct) / float(num_samples) * 100:.2f}%')
    model.train()
    return float(num_correct)/float(num_samples)


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

print(loss_list)
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()






'''
model = NN(input_size=784, num_classes=10)
x = torch.randn(64, 784)
print(model(x).shape)
'''

