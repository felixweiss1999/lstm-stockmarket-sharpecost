from Random_Buy_Example import *
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import random

# Define the LSTM model class
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) #input dim: batch size, seqlength, inputsize
        self.fc = nn.Linear(hidden_size, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x): #x: (batch, seq, features) tensor 
        #initial hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0)) #out is (batch, seq, hidden_size)
        out = self.fc(out[:,-1,:]) #decode only last outputs of each sequence!
        # out is (batch, output_dim)
        out = self.softmax(out)
        return out

class pastReturnsContainer:
    def __init__(self, n):
        self.buffer = [torch.Tensor([0.0])] * n
        self.index = 0
        #self.sum = 0
        self.count = 0
    
    def add_detach(self, x):
        #self.sum -= self.buffer[self.index]
        
        self.buffer[self.index] = x.detach() #completely detaches the gradients stored in this list from gradient computation
        #self.sum += x
        self.index = (self.index + 1) % len(self.buffer)
        self.count = min(self.count + 1, len(self.buffer))

    def get_sum(self):
        return torch.sum(torch.Tensor(self.buffer))
    def get_count(self):
        return self.count
def generateBatch(data, batchsize, seqlength, offset): #expects sequence to go along first dimenstion of data
    x = []
    y = []
    for i in range(batchsize):
        x.append(data[offset + i:offset + i+seqlength][:])
        y.append(data[offset + i+seqlength][:])
    x = torch.Tensor(np.array(x))
    y = torch.Tensor(np.array(y))
    return x, y

def average_array(original_array, new_length):
    step_size = original_array.shape[0] // new_length
    averaged_array = np.zeros(new_length)
    for i in range(new_length):
        start_index = i * step_size
        end_index = (i + 1) * step_size
        averaged_array[i] = np.mean(original_array[start_index:end_index])
    return averaged_array

def generateBatches(data, batchsize, seqlength, shuffle : bool):
    x = []
    y = []
    s = ceil((len(data) - seqlength) / batchsize) - 1
    for i in range(s):
        xbatch = []
        ybatch = []
        for j in range(min(batchsize, (len(data) - seqlength) - batchsize * i)):
            xbatch.append(np.ndarray.tolist(data[(i*batchsize) + j : (i*batchsize) + j+seqlength]))
            ybatch.append(np.ndarray.tolist(data[i*batchsize + j + seqlength]))
        x.append(xbatch)
        y.append(ybatch)
    assert len(x) == len(y)
    if shuffle:
        indices = random.sample(range(len(x)), len(x))
        x = [x[i] for i in indices]
        y = [y[i] for i in indices]
    return torch.Tensor(x), torch.Tensor(y)

def sharpeCost(future, allocRatios, pastReturns : pastReturnsContainer, pastReturnsSquared : pastReturnsContainer):
    currentReturns = torch.sum(future * allocRatios, dim=1)
    e = []
    eInnerSquared = []
    for ret in currentReturns: #按照順序去計算各個batch裏面的成本!
        e.append(torch.div(pastReturns.get_sum() + ret, pastReturns.get_count() + 1).unsqueeze(0))
        pastReturns.add_detach(ret)
        eInnerSquared.append(torch.div(pastReturnsSquared.get_sum() + torch.pow(ret, 2), pastReturnsSquared.get_count() + 1).unsqueeze(0))
        pastReturnsSquared.add_detach(torch.pow(ret,2))
    e = torch.cat(e)
    eInnerSquared = torch.cat(eInnerSquared)
    esquared = torch.pow(e, 2)
    return torch.mean(torch.mul(torch.div(e, torch.sqrt(torch.sub(eInnerSquared, esquared))), -1))



readIndividualStocksIntoLargeFiles(list=[2330, 2317, 2454, 2412, 6505, 2308, 2881, 2882, 1303, 1301])
open = np.array(evenOutData(extractFeature('open', loadStocks('samples//customsamples.json')))).transpose()
close = np.array(evenOutData(extractFeature('close', loadStocks('samples//customsamples.json')))).transpose()
open, close, removed = remove_rows_with_zeros(open, close)
changeDat = (close / open) - 1



model = LSTM(input_size=changeDat.shape[1], hidden_size=64, num_layers=1, output_dim=changeDat.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)

rollingReturns = pastReturnsContainer(49)
rollingReturnsSquared = pastReturnsContainer(49)


model.train()
for epoch in range(100):
    x, y = generateBatches(changeDat, batchsize=64, seqlength=50, shuffle=True)
    for i, batch in enumerate(x):
        optimizer.zero_grad()
        out = model(batch)
        testloss = sharpeCost(y[i], out, rollingReturns, rollingReturnsSquared)
        if(i>0):
            testloss.backward()
            optimizer.step()


model.eval()
x,_ = generateBatch(changeDat, 1, 50, 1000)
print(model(x))
torch.save(model.state_dict(), 'model.pth')
model = LSTM(input_size=changeDat.shape[1], hidden_size=64, num_layers=1, output_dim=changeDat.shape[1])
print(model(x))
model.load_state_dict(torch.load('model.pth'))
model.eval()
print(model(x))

