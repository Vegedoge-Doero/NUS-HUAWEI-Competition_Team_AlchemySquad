%% correlation between days
clc; clear all;
%% Sample Stocks
filepath_sample = 'SampleStocks.csv';
symbols = readtable(filepath_sample).Code;
%% read all the data
clc;
file_list = dir("./dataset_json_priceList");
files = {file_list(3:end).name};
for i = 1:99
    fileName = files{i};
    str = fileread("./dataset_json_priceList/"+fileName); % Opening the file
    data(i) = jsondecode(str); % Using the jsondecode function to parse JSON from string
end
%%
for i = 100:length(files)
    fileName = files{i};
    str = fileread("./dataset_json_priceList/"+fileName); % Opening the file
    data2023(i-99) = jsondecode(str); % Using the jsondecode function to parse JSON from string
end
%%
clc;
correlation_length = zeros(99,length(symbols));
start = 1;
last = 99;
for i = start:last
    for j = 1:length(symbols)
        price1 = data(1).(strrep("x"+symbols(j),'.','_'));
        price1 = rescale(imresize(price1,[4000,1]),0,1);
        price2 = data(i).(strrep("x"+symbols(j),'.','_'));
        price2 = rescale(imresize(price2,[4000,1]),0,1);
        correlation_length(i, j) = correlation_length(i, j) + (price1'*price2)/4000 - mean(price1)*mean(price2);
    end
end
%%
correlation_length1 = zeros(20,length(symbols));
start = 1;
last = 20;
for i = start:last
    for j = 1:length(symbols)
        price1 = data2023(1).(strrep("x"+symbols(j),'.','_'));
        price1 = rescale(imresize(price1,[4000,1]),0,1);
        price2 = data2023(i).(strrep("x"+symbols(j),'.','_'));
        price2 = rescale(imresize(price2,[4000,1]),0,1);
        correlation_length1(i, j) = correlation_length1(i, j) + (price1'*price2)/4000 - mean(price1)*mean(price2);
    end
end
colormap1 = [[16/256, 70/256, 128/256];...
            [27/256, 95/256, 155/256];...
            [49/256, 124/256, 183/256];...
            [79/256, 150/256, 196/256];...
            [109/256, 173/256, 209/256];...
            [140/256, 194/256, 220/256];...
            [182/256, 215/256, 232/256];...
            [205/256, 225/256, 238/256];...
            [233/256, 241/256, 244/256];...
            [242/256, 235/256, 230/256];...
            [251/256, 227/256, 213/256];...
            [248/256, 200/256, 180/256];...
            [246/256, 178/256, 147/256];...
            [232/256, 140/256, 117/256];...
            [220/256, 109/256, 87/256];...
            [200/256, 69/256, 62/256];...
            [183/256, 34/256, 48/256];...
            [140/256, 15/256, 40/256];...
            [109/256, 1/256, 31/256]];
%%
imagesc(0:100, 0:100, correlation_length');colormap jet;colorbar('LineWidth',1.1);
xlabel("Date"); ylabel("Index of the stocks");
set(gca,'FontName','Arial','FontSize',18,'LineWidth',1.1);


