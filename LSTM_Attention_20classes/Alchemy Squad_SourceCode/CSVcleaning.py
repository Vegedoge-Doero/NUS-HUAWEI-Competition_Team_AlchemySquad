import os
import pandas as pd
import json
import numpy as np
from testStocks import get_ms
dir_list = os.listdir("../dataset")
dir_new_list = os.listdir("../dataset_cleaned")

for jj in range(len(dir_list)):
    tick_data = pd.read_csv("../dataset_cleaned/"+dir_list[jj], index_col=None,
                            usecols=['COLUMN02', 'COLUMN03', "COLUMN07", "COLUMN48", "COLUMN49"])
    # tick_data.to_csv("../dataset_cleaned/"+dir_list[i])
    symbol = list(tick_data['COLUMN02'].unique())
    symbol.sort()  # stocks' name
    priceList = {}
    msList = {}
    VWAP = {}
    i = 0
    for sym in symbol:
        sym_data = tick_data[tick_data['COLUMN02'] == sym]
        ms = sym_data['COLUMN03'].apply(lambda x: get_ms(x)).values
        price = sym_data['COLUMN07'].values
        priceList[sym] = np.array(price).tolist()
        msList[sym] = np.array(ms).tolist()
        VWAP[sym] = sym_data.iloc[-1]['COLUMN49'] / sym_data.iloc[-1]['COLUMN48']
        i = i + 1
    with open("../dataset_json_priceList/priceList"+dir_list[jj]+".json", "w") as fp:
        json.dump(priceList, fp)
    with open("../dataset_json_msList/msList"+dir_list[jj]+".json", "w") as fp:
        json.dump(msList, fp)
    with open("../dataset_json_VWAP/VWAP"+dir_list[jj]+".json", "w") as fp:
        json.dump(VWAP, fp)
    with open("../dataset_json_stock_name/stock_name"+dir_list[jj]+".json", "w") as fp:
        json.dump(symbol, fp)
    print("Finished "+dir_list[jj])
