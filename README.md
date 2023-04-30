# NUS-HUAWEI-Competition
This is the code repository of the NUS-Huawei Competition. Rules, trading procedure, and data can be found on the official competition website: https://www.sg-innovationchallenge.org/

The programs contain several parts. Taking the LSTM_Attention_20classes/Alchemy Squad_SourceCode/ folder as an example, they are:
1. Neural network schemes (NNclass.py);
2. Data cleaning (CSVcleaning.py);
3. Model training (myTrainModel_json.py);
4. Real-time trading process (myModel_test_multiple_20230317.py) with pre-trained model (CSV99_LSTM_class20_checkpoint200.pth.tar);
5. Evaluation (evaluation_multiple_20230317.py in the root path)

Training and test data can be found on the competition website: https://e-share.obs-website.cn-north-1.myhuaweicloud.com/?token=hEV8oNaCGcu2ERP24m0Qyxmfvtc6ep5nw3hLioUvQ9CiQ7v3GiFV7sJK/9TYoyL/G5nKe47cUZJiuQWxVsdhi3rKwNID8AuNtJrBmo6DTHYTasHb1ph7QVCEv0Ag6qs5nZeo7G+VfWrp9BWe2+oXAQAEbzDR8vmrS2P4Zmci6Vonin6DM/x18gLoM9hK1+OEYf1v/XJorNoX+o3w0GrI6iDCb2QYFxDODPqw54zqBJtz9vL0Y9Q+I87/+rNj6fvP+XJHLrf6Bb+esIZiRKqstG0KVQFh55eQ/7HRwZP5rovVsj8yaaWD8r13PEmiJzy4JwVOnkM/LeZTibSV9aJcbXphh4rHLvTkIHfXlHKAy5eqbedK+uzM0WZeGD9lilVC/EcoL5M6CTKm1ux+a2FWDDBNv35dyy89TQ0FcoXU0pJZg1oCXFnLfUvw8+WixOs4hJgcqwJruSsXxxk2ym3MwtDkXYt13al/Evojq4sERi74Nx6QTUtag1PRsEqa4H8PeXoq0oyhLD+FnoQTTF1YYJsvq0THBzehY42zhWSBiLoQhSu9mSrxWk1hM5c9NUqI/k8JtxiVeDTFz4HZDXYVDl5N+ozqeci0l9in2LDW/Q+X+FUHCPTFx5X+MX5NM7YCj1Ksf7v3qDJ7DZZvCdNZeA. The extraction code is: nusbac. CSVcleaning.py can filter out useful elements such as time, tickname, price, VWAP, and convert them from csv to json format.

To execute the program, the model needs to be trained in the first place. The program will automatically save the pre-trained network's parameters into pth file. Then the real-time trading code can be executed. In this part, 100 stocks' single-day real-time data will be sent (FIFO) into the program. The program will write the decision (Buy: B, Sell: S, Nothing: N) into a CSV file, for every time and every stock. Lastly, executing the evaluation program will provide the earning rate (in bp) for each day. 

There are two versions of the trading model: LSTM + Attention + 20 classes, and LSTM + 10 classes. In addition, RNN, BLSTM, CNN, GRU have been explored (structures of them can be accessed in NNclass.py). Users can execute these models as well by changing the useNN variable in the model training code. The detailed exploration, analysis, and results comparisons are summarized in the file of Explaination_Deck.pptx (presentation video can be ). The code of data analysis and results comparisons can be found in the data_analysis folder.


