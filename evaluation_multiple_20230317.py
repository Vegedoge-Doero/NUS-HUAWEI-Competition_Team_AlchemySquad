#!/usr/bin/env python 
# -*- coding:utf-8 -*

import pandas as pd
import numpy as np
import sys



def tm_to_ms(tm):
    hhmmss = tm // 1000
    ms = (hhmmss // 10000 * 3600 + (hhmmss // 100 % 100) * 60 + hhmmss % 100) * 1000 + tm % 1000
    return ms


def check_validity(df):
    if not df.shape[0] >= 3:
        return 'number of orders of a single stock must be not less than 3'
    if not np.array_equal(df['volume'], df['volume'].astype(int)):
        return 'field VOLUME is not in integer format'
    if not (df['volume'] >= 1).all():
        return 'the volume of each transaction must be not less than 1'
    if not df['volume'].sum() == 100:
        print(df['volume'])
        print(df['volume'].sum())
        return 'the total volume of a single stock must be not larger than 100'
    if not df['tickTm'].apply(tm_to_ms).diff().min() > 60000:
        return 'interval between two consecutive transactions must be not less than 1 minute'
    return None

useNN = 'LSTM'
filedates=["0805", "0808", "0809", "0810", "0811", "0812", "0815", "0816", "0817", "0818", "0819",
           "0822", "0823", "0824", "0825", "0826", "0829", "0830", "0831", "0901", "0902", "0905",
           "0906", "0907", "0908", "0909", "0913", "0914", "0915", "0916", "0919", "0920", "0921",
           "0922", "0923", "0926", "0927", "0928", "0929", "0930", "1010", "1011", "1012", "1013",
           "1014", "1017", "1018", "1019", "1020", "1021", "1024", "1025", "1026", "1027", "1028",
           "1031", "1101", "1102", "1103", "1104", "1107", "1108", "1109", "1110", "1111", "1114",
           "1115", "1116", "1117", "1118", "1121", "1122", "1123", "1124", "1125", "1128", "1129",
           "1130", "1201", "1202", "1205", "1206", "1207", "1208", "1209", "1212", "1213", "1214",
           "1215", "1216", "1219", "1220", "1221", "1222", "1223", "1226", "1227", "1228", "1229"]
filedates = ["0306", "0307", "0308", "0309", "0310", "0313", "0314", "0315", "0316", "0317", "0320", "0321", "0322",
             "0323", "0324", "0327", "0328", "0329", "0330", "0331"]
symbol_file = '../SampleStocks.csv'
symbol = pd.read_csv(symbol_file, index_col=None)['Code'].to_list()

for filedate in filedates[0:]:
    tick_data_file = f"../dataset/tickdata_2023{filedate}.csv"
    order_time_file = f"./output_file/outputMy{useNN}{filedate}.csv"
    tick_data = pd.read_csv(tick_data_file, index_col=None)
    order_data = pd.read_csv(order_time_file, index_col=None)
    # Order Sequence Check
    if not np.array_equal(order_data['dataIdx'], tick_data['COLUMN01']):

        mismatchIdx = [i for i in range(order_data.shape[0]) if order_data['dataIdx'][i] != tick_data['COLUMN01'][i]]
        print('order-tick index mismatch: {}, {}'.format(order_data['dataIdx'][mismatchIdx[0]],
                                                     tick_data['COLUMN01'][mismatchIdx[0]]))
        sys.exit()

    order_data = order_data[order_data['BSflag'] != 'N']
    tick_tm = tick_data['COLUMN03'].tolist()
    order_data['tickTm'] = [tick_tm[i] for i in order_data['dataIdx'].tolist()]
    tick_prc = tick_data['COLUMN07'].tolist()
    order_data['tickPrc'] = [tick_prc[i] for i in order_data['dataIdx'].tolist()]


    profit = []
    for sym in symbol:
        sym_data = tick_data[tick_data['COLUMN02'] == sym]
        mkt_mean_prc = sym_data.iloc[-1]['COLUMN49'] / sym_data.iloc[-1]['COLUMN48']

        sym_order_buy = order_data[(order_data['symbol'] == sym) & (order_data['BSflag'] == 'B')]
        errInfo = check_validity(sym_order_buy)
        if errInfo:
            print(f'Buy of {sym} invalid: {errInfo}')
            sys.exit()
        buy_mean_prc = (sym_order_buy['volume'] * sym_order_buy['tickPrc']).sum() / 100

        sym_order_sell = order_data[(order_data['symbol'] == sym) & (order_data['BSflag'] == 'S')]
        errInfo = check_validity(sym_order_sell)
        if errInfo:
            print(f'Sell of {sym} invalid: {errInfo}')
            sys.exit()
        sell_mean_prc = (sym_order_sell['volume'] * sym_order_sell['tickPrc']).sum() / 100
        profit.append((sell_mean_prc - buy_mean_prc) / mkt_mean_prc)

    print(f'{filedate}: ', 'Earning rate is {:.2f} bp'.format(np.mean(profit)))
