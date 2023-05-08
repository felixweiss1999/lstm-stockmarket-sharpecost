import random, torch, json
import torch.nn as nn
import numpy as np
from datetime import datetime
from Random_Buy_Example import *


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) #input dim: batch size, seqlength, inputsize
        self.fc = nn.Linear(hidden_size, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x): #put in (batch, seq, features) tensor as x
        #initial hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0)) #out is (batch, seq, hidden_size)
        out = self.fc(out[:,-1,:]) #decode only last outputs of each sequence!
        # out is (batch, output_dim)
        out = self.softmax(out)
        return out

account = 'p76118400'
password = '18400'
stockidlist = [2330, 2317, 2454, 2412, 6505, 2308, 2881, 2882, 1303, 1301]
today = datetime.today().strftime('%Y%m%d')

#get data
user_stocks = Get_User_Stocks(account, password) 
dat = []
for i in stockidlist:
    dat.append(Get_Stock_Informations(i, "20220222", today))

open = np.array(evenOutData(extractFeature('open', dat))).transpose()
close = np.array(evenOutData(extractFeature('close', dat))).transpose()
open, close, removed = remove_rows_with_zeros(open, close)

changeDat = (close / open) - 1
torch.set_printoptions(sci_mode=False)

x = torch.Tensor(np.array([changeDat[-51:-1]]))

#load model, make portfolio_weights
model = model = LSTM(input_size=changeDat.shape[1], hidden_size=64, num_layers=1, output_dim=changeDat.shape[1])
print(model(x))
model.load_state_dict(torch.load('goodModelWithoutTransactionCost.pth'))
model.eval()
portfolio_weights = model(x)
print(portfolio_weights)

#buy/sell stocks accordingly

#1 determine current net_worth
net_worth = 51149
idToAmountDict = {}
for stock in user_stocks:
    print('adding stock id', int(stock['stock_code_id']), 'at list index', stockidlist.index(int(stock['stock_code_id'])), 'worth', close[-1][stockidlist.index(int(stock['stock_code_id']))], '*', stock['shares'])
    net_worth += close[-1][stockidlist.index(int(stock['stock_code_id']))] * stock['shares'] #current price * amount
    idToAmountDict[int(stock['stock_code_id'])] = stock['shares']
print("networth:",net_worth)
print(idToAmountDict)
input()

high = np.array(evenOutData(extractFeature('high', dat))).transpose()
low = np.array(evenOutData(extractFeature('low', dat))).transpose()
for i, stockid in enumerate(stockidlist):
    idealAmount = int((net_worth/close[-1][i]) * portfolio_weights[0][i])
    if stockid in idToAmountDict:
        currentAmount = idToAmountDict[stockid]
    else:
        currentAmount = 0
    amountChange = idealAmount - currentAmount

    if(amountChange > 0): #need to buy
        random_trading_price_offset_range = high[-1][i] - low[-1][i]
        buy_price = close[-1][i] - random.random() * random_trading_price_offset_range
        print('Recommendation: Buy ', amountChange, 'x stock with ID', stockid, 'for', buy_price, '$ a piece. The new percentage taken up by this stock would be', portfolio_weights[0][i]*100,'percent of current networth', net_worth,'$.')
        input()
        Buy_Stock(account, password, stockid, amountChange, buy_price)
    elif(amountChange < 0):
        amountChange *= -1
        random_trading_price_offset_range = high[-1][i] - low[-1][i]
        sell_price = close[-1][i] + random.random() * random_trading_price_offset_range
        print('Recommendation: Sell ', amountChange, 'x stock with ID', stockid, 'for', sell_price, '$ a piece. The new percentage taken up by this stock would be', portfolio_weights[0][i]*100,'percent of current networth', net_worth,'$.')
        input()
        Sell_Stock(account, password, stockid, amountChange, sell_price)





