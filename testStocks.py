import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import sys

size0 = 10
size1 = 50

def get_ms(tm):
    hhmmss = tm // 1000
    ms = (hhmmss // 10000 * 3600 + (hhmmss // 100 % 100) * 60 + hhmmss % 100) * 1000 + tm % 1000
    ms_from_open = ms - 34200000  # millisecond from stock opening
    if tm >= 130000000:
        ms_from_open -= 5400000
    return ms_from_open


def pickData(input_file):
    tick_data = pd.read_csv(input_file, index_col=None)
    symbol = list(tick_data['COLUMN02'].unique())
    symbol.sort()  # stocks' name
    priceList = {}
    msList = {}
    VWAP = {}
    stock_name = {}
    i = 0
    for sym in symbol:
        sym_data = tick_data[tick_data['COLUMN02'] == sym]
        ms = sym_data['COLUMN03'].apply(lambda x: get_ms(x)).values
        price = sym_data['COLUMN07'].values
        priceList[sym] = np.array(price)
        msList[sym] = np.array(ms)
        VWAP[sym] = sym_data.iloc[-1]['COLUMN49'] / sym_data.iloc[-1]['COLUMN48']
        stock_name[i] = sym
        i = i+1
    return priceList, msList, VWAP, stock_name


class StocksTensor(Dataset):
    def __init__(self, size, window, priceList, msList, VWAP, stock_name, transform=None, train=True):
        self.priceList = priceList
        self.msList = msList
        self.VWAP = VWAP
        self.stock_name = stock_name
        self.size = size
        self.window = window
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        index = index % len(self.stock_name)
        length = int(np.random.randint(low=size0*size1, high=len(self.priceList[self.stock_name[index]]), size=1))
        x = np.array(self.priceList[self.stock_name[index]])[length-size0*size1:length]
        x0 = np.array(self.priceList[self.stock_name[index]])[0]
        x00 = x[0]
        x_mean = np.average(np.array(self.priceList[self.stock_name[index]])[0:length])
        #normalization = np.max(x)-np.min(x)
        #print(normalization)
        stock_label = (x[-1] - 10000*self.VWAP[self.stock_name[index]]) / x0  #/ normalization
        stock_label = stock_label.astype('float32')
        x = np.reshape(x, (size0, size1, 1))
        x = (x - x_mean) / x0 * 1000  # normalization
        x = x.astype('float32')
        y_label = torch.tensor([5])
        if stock_label > 0.00:
            y_label = torch.tensor([5])
        if stock_label > 0.0013:
            y_label = torch.tensor([6])
        if stock_label > 0.0031:
            y_label = torch.tensor([7])
        if stock_label > 0.0055:
            y_label = torch.tensor([8])
        if stock_label > 0.0094:
            y_label = torch.tensor([9])
        if stock_label < -0.00:
            y_label = torch.tensor([4])
        if stock_label < -0.002:
            y_label = torch.tensor([3])
        if stock_label < -0.0039:
            y_label = torch.tensor([2])
        if stock_label < -0.0062:
            y_label = torch.tensor([1])
        if stock_label < -0.0101:
            y_label = torch.tensor([0])
        if self.transform:
            x = self.transform(x)
        return x, y_label


#priceList, msList, VWAP, stock_name = pickData(input_file="./dataset/tickdata_20220805.csv")
