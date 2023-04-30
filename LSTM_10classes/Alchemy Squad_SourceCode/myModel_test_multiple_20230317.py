import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import sys
from testStocks import StocksTensor, pickData
from NNclass import *
import math
import joblib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

size0 = 10
size1 = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 100
num_classes = 10
learning_rate = 0.01
batch_size = 64
in_channels = 1  # for CNN
sequence_length = 10  # for RNN
input_size_RNN = 50  # for RNN
num_layers = 2  # for RNN
hidden_size = 100  # for RNN
size = 1  # for random dataset

# ---------- Initialization ----------
# Initialize
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
    model = LSTM(input_size=input_size_RNN, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes,
                 device=device).to(device)
elif useNN == 'BLSTM':
    model = BLSTM(input_size=input_size_RNN, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes,
                  device=device).to(device)
elif useNN == 'VGG16':
    model = torchvision.models.vgg16(pretrained=False)
    model.features[0] = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                  padding=(1, 1))
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


def get_ms(tm):
    hhmmss = tm // 1000
    ms = (hhmmss // 10000 * 3600 + (hhmmss // 100 % 100) * 60 + hhmmss % 100) * 1000 + tm % 1000
    ms_from_open = ms - 34200000  # millisecond from stock opening
    if tm >= 130000000:
        ms_from_open -= 5400000
    return ms_from_open


def check_accuracy(x, model):
    model.eval()
    with torch.no_grad():
        x = x.to(device=device)
        if useNN == 'NN':
            x = x.reshape(x.shape[0], -1)
        if useNN in ['RNN', 'GRU', 'LSTM', 'BLSTM']:
            x = x.squeeze(1)
        scores = model(x)
    model.train()
    return scores


loadpoint = 200
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
load_checkpoint(torch.load(f'./{useNN}_checkpoint{loadpoint}.pth.tar'), model, optimizer)
print('Load model successfully!')
symbol_file = './SampleStocks.csv'
symbol = pd.read_csv(symbol_file, index_col=None)['Code'].to_list()
idx_dict = dict(zip(symbol, list(range(len(symbol)))))


# Multiple files processing
for filedate in range(1):
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    tick_data = open(input_path, 'r')
    order_time = open(output_path, 'w')
    target_vol = 100
    basic_vol = 2
    cum_vol_buy = [0] * len(symbol)  # accumulate buying volume
    cum_vol_sell = [0] * len(symbol)  # accumulate selling volume
    unfinished_buy = [target_vol] * len(symbol)  # unfinished buying volume in current round
    unfinished_sell = [target_vol] * len(symbol)  # unfinished selling volume in current round
    last_buy_ms = [0] * len(symbol)  # last buy time
    last_sell_ms = [0] * len(symbol)  # last sell time
    hist_ms = [[] for i in range(len(symbol))]  # historic time
    hist_prc = [[] for i in range(len(symbol))]  # historic price
    buy_position = [[] for i in range(len(symbol))]  # record buying positions
    sell_position = [[] for i in range(len(symbol))]  # record sell positions
    buy_volume = [[] for i in range(len(symbol))]  # record buying volumes
    sell_volume = [[] for i in range(len(symbol))]  # record sell volumes
    buy_price = [[] for i in range(len(symbol))]  # record buying prices
    sell_price = [[] for i in range(len(symbol))]  # record sell prices
    # --------------- Loop ---------------
    # recursively read all tick lines from tickdata file,
    # do decision with your strategy and write order to the ordertime file

    tick_data.readline()  # header
    order_time.writelines('symbol,BSflag,dataIdx,volume\n')
    order_time.flush()

    while True:
        tick_line = tick_data.readline()  # read one tick line
        if tick_line.strip() == 'stop' or len(tick_line) == 0:
            break
        row = tick_line.split(',')
        nTick = row[0]
        if int(nTick) % 100000 == 0:
            print(f'At the position {nTick} of the CSV-{filedate}')
        sym = row[1]
        tm = int(row[2])
        if sym not in symbol:
            order_time.writelines(f'{sym},N,{nTick},0\n')
            order_time.flush()
            continue

        # -------- Your Strategy Code Begin --------
        idx = idx_dict[sym]
        tm_ms = get_ms(tm)
        prc = int(row[6])
        hist_ms[idx].append(tm_ms)
        hist_prc[idx].append(prc)
        order = ('N', 0)
        startpoint = 500  # wait until this point, we start to do transactions
        current_point = len(hist_prc[idx])
        if current_point >= startpoint:
            x = np.array(hist_prc[idx][current_point-startpoint:current_point])
            normalization = np.max(x)
            x0 = hist_prc[idx][0]  # open price
            x00 = x[0]  # last price 500 points before
            x_avg = np.average(hist_prc[idx][0:current_point])  # average price since open
            x = (x - x_avg) / x0 * 1000
            x = np.reshape(x, (1, 1, size0, size1))
            x = x.astype('float32')
            x = torch.tensor(x)  # input
            scores = check_accuracy(x, model)  # output loss, output probability
            _, predictions = scores.max(1)
            order = ('N', 0)  # initialize the order

            if tm_ms < 10800000:  # before 14:00:00 (10800s = 180min = 3h)
                if predictions > 5 and tm_ms - last_sell_ms[idx] > 75000 and cum_vol_sell[idx] < 100:
                    sell_vol = predictions - 4
                    if sell_vol > 100 - cum_vol_sell[idx]:
                        sell_vol = 100 - cum_vol_sell[idx]
                    order = ('S', sell_vol)
                    sell_position[idx].append(current_point)
                    sell_volume[idx].append(sell_vol)
                    sell_price[idx].append(hist_prc[idx][current_point-1])
                    last_sell_ms[idx] = tm_ms
                    cum_vol_sell[idx] += sell_vol
                    unfinished_sell[idx] -= sell_vol
                if predictions < 4 and tm_ms - last_buy_ms[idx] > 75000 and cum_vol_buy[idx] < 100:
                    buy_vol = 5 - predictions
                    if buy_vol > 100 - cum_vol_buy[idx]:
                        buy_vol = 100 - cum_vol_buy[idx]
                    order = ('B', buy_vol)
                    buy_position[idx].append(current_point)
                    buy_volume[idx].append(buy_vol)
                    buy_price[idx].append(hist_prc[idx][current_point-1])
                    last_buy_ms[idx] = tm_ms
                    cum_vol_buy[idx] += buy_vol
                    unfinished_buy[idx] -= buy_vol


            else:  # force complete before market closes
                if cum_vol_sell[idx] < 100 or cum_vol_buy[idx] < 100:
                    if tm_ms - last_buy_ms[idx] > 60000 and cum_vol_buy[idx] < 100:
                        print(f'Reach to 14:50:00. Force to finish {sym}. '
                              f'{int(100 - cum_vol_sell[idx])} left to sell, and {int(100 - cum_vol_buy[idx])} left to buy.')
                        splitter = 1 if len(buy_position[idx]) >= 2 else 3 - len(buy_position[idx])
                        order = ('B', int((100-cum_vol_buy[idx])/splitter))
                        last_buy_ms[idx] = tm_ms
                        buy_position[idx].append(current_point)
                        buy_volume[idx].append(int((100-cum_vol_buy[idx])/splitter))
                        cum_vol_buy[idx] += int((100-cum_vol_buy[idx])/splitter)
                        unfinished_buy[idx] -= int((100-cum_vol_buy[idx])/splitter)
                        buy_price[idx].append(hist_prc[idx][current_point-1])
                    elif tm_ms - last_sell_ms[idx] > 60000 and cum_vol_sell[idx] < 100:
                        print(f'Reach to 14:50:00. Force to finish {sym}. '
                              f'{int(100 - cum_vol_sell[idx])} left to sell, and {int(100 - cum_vol_buy[idx])} left to buy.')
                        splitter = 1 if len(sell_position[idx]) >= 2 else 3 - len(sell_position[idx])
                        order = ('S', int((100-cum_vol_sell[idx])/splitter))
                        last_sell_ms[idx] = tm_ms
                        sell_position[idx].append(current_point)
                        sell_volume[idx].append(int((100-cum_vol_sell[idx])/splitter))
                        cum_vol_sell[idx] += int((100-cum_vol_sell[idx])/splitter)
                        unfinished_sell[idx] -= int((100-cum_vol_sell[idx])/splitter)
                        sell_price[idx].append(hist_prc[idx][current_point-1])

            # write order
            if order[0] == 'N':
                order_time.writelines(f'{str(sym)},N,{nTick},0\n')
                order_time.flush()
            else:
                order_time.writelines(f'{str(sym)},{order[0]},{nTick},{int(order[1])}\n')
                order_time.flush()
        else:
            order_time.writelines(f'{str(sym)},N,{nTick},0\n')
            order_time.flush()

        # -------- Your Strategy Code End --------

    # ---------- Post Processing ----------

    tick_data.close()
    order_time.close()
    draw_decision = False
    if draw_decision:
        fig, axs = plt.subplots(5, 5)
        for count in range(100):
            buy_volume1 = torch.tensor(buy_volume[count]).detach().cpu()
            sell_volume1 = torch.tensor(sell_volume[count]).detach().cpu()
            buy_price1 = torch.tensor(buy_price[count]).detach().cpu()
            sell_price1 = torch.tensor(sell_price[count]).detach().cpu()
            earn = (np.dot(np.array(sell_volume1), np.array(sell_price1)) -
                    np.dot(np.array(buy_volume1), np.array(buy_price1))) / np.mean(np.array(hist_prc[count]))
            axs[math.floor(count / 5) % 5, count % 5].plot(np.array(hist_prc[count]), c=(0, 0, 1))
            axs[math.floor(count / 5) % 5, count % 5].scatter(np.array(buy_position[count]),
                                                              np.array(buy_price[count]),
                                                              s=np.array(buy_volume1)*10, c=(0, 1, 0))
            axs[math.floor(count / 5) % 5, count % 5].scatter(np.array(sell_position[count]),
                                                              np.array(sell_price[count]),
                                                              s=np.array(sell_volume1)*10, c=(1, 0, 0))
            axs[math.floor(count / 5) % 5, count % 5].set_title(f'{symbol[count]} Earning {earn:.2f} bp')
            if count % 25 == 24:
                plt.show()
                fig, axs = plt.subplots(5, 5)
