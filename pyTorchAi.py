from Random_Buy_Example import *
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import random
#torch.manual_seed(42)
# Define the LSTM model class
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
            # print('added',(i*batchsize) + j,':',(i*batchsize) + j+seqlength, 'to batch',i)
            # input()
        x.append(xbatch)
        y.append(ybatch)
    # print(len(x), len(x[0]), len(x[0][0]))
    # print(len(y), len(y[0]))
    assert len(x) == len(y)
    if shuffle:
        indices = random.sample(range(len(x)), len(x))
        x = [x[i] for i in indices]
        y = [y[i] for i in indices]
    return torch.Tensor(x), torch.Tensor(y)

def sharpeCost(future, allocRatios, pastReturns : pastReturnsContainer, pastReturnsSquared : pastReturnsContainer):
    #calculate new net gain based on model allocation ratios
    currentReturns = torch.sum(future * allocRatios, dim=1) #performs elementwise multiplication for all elements in the batches.
    #print(currentReturns)

    e = []
    eInnerSquared = []
    for ret in currentReturns: #for each element in batch
        e.append(torch.div(pastReturns.get_sum() + ret, pastReturns.get_count() + 1).unsqueeze(0))
        pastReturns.add_detach(ret)
        eInnerSquared.append(torch.div(pastReturnsSquared.get_sum() + torch.pow(ret, 2), pastReturnsSquared.get_count() + 1).unsqueeze(0))
        pastReturnsSquared.add_detach(torch.pow(ret,2))
    e = torch.cat(e)
    eInnerSquared = torch.cat(eInnerSquared)
    #print(e)
    
    esquared = torch.pow(e, 2)
    
    return torch.nanmean(torch.mul(torch.div(e, torch.sqrt(torch.sub(eInnerSquared, esquared))), -1))





#readIndividualStocksIntoLargeFiles(list=[2603,2330, 2308])
readIndividualStocksIntoLargeFiles(list=[2330, 2317, 2454, 2412, 6505, 2308, 2881, 2882, 1303, 1301])
open = np.array(evenOutData(extractFeature('open', loadStocks('samples//customsamples.json')))).transpose()
close = np.array(evenOutData(extractFeature('close', loadStocks('samples//customsamples.json')))).transpose()
open, close, removed = remove_rows_with_zeros(open, close)

changeDat = (close / open) - 1
print(len(changeDat))


model = LSTM(input_size=changeDat.shape[1], hidden_size=64, num_layers=1, output_dim=changeDat.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)

rollingReturns = pastReturnsContainer(49)
rollingReturnsSquared = pastReturnsContainer(49)

torch.autograd.set_detect_anomaly(True)
losses = []
averages = []
modeloutputs = []
model.train()
torch.set_printoptions(sci_mode=False)
for epoch in range(100):
    x, y = generateBatches(changeDat, batchsize=64, seqlength=50, shuffle=True)
    losses.append([])
    modeloutputs.append([])
    for i, batch in enumerate(x):
        optimizer.zero_grad()
        #x, y = generateBatch(changeDat, batchsize=1, seqlength=50, offset=iter)
        out = model(batch)

        #print(out)
        #input()
        #print(y)
        #print(out)
        #rollingReturns = pastReturnsContainer(49)
        #rollingReturnsSquared = pastReturnsContainer(49)
        testloss = sharpeCost(y[i], out, rollingReturns, rollingReturnsSquared)
        #print(testloss) #todo: testloss muss aufaddiert werden, und erste iteration muss geskipped werden
        
        if(i>0):
            testloss.backward() #calculate gradients
            optimizer.step() #update weights
            losses[epoch].append(testloss.item() * -1)
            modeloutputs[epoch].append(out)
    averages.append(np.mean(losses[epoch]))
    print(epoch)
np.savetxt('averageLosses.txt', averages)
np.savetxt("losses.txt", losses)


model.eval()
x,_ = generateBatch(changeDat, 1, 50, 1000)

print(model(x))

print(averages)
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(list(range(len(losses[0]))), losses[0], 'r', label='0')
#ax1.plot(list(range(len(losses[4]))), losses[4], 'g', label='1')
lastelem = len(losses)-1
ax1.plot(list(range(len(losses[lastelem]))), losses[lastelem], 'b', label='2')
ax2 = ax1.twinx()
ax2.plot(list(range(len(losses[lastelem]))), average_array(np.mean(changeDat, axis=1), len(losses[lastelem])), 'black')
#ax3 = fig.add_subplot(212)
#for i in range(len(modeloutputs[0][0])):
#    ax3.plot(torch.arange(len(modeloutputs[0][0])), modeloutputs[0][i].detach().numpy(), label=f'Tensor {i}')

# fig2 = plt.figure()
# for i in range(len(modeloutputs[0][0])):
#     plt.plot([t[i] for t in modeloutputs[0]])
plt.show()

torch.save(model.state_dict(), 'model.pth')
model = LSTM(input_size=changeDat.shape[1], hidden_size=64, num_layers=1, output_dim=changeDat.shape[1])
print(model(x))
model.load_state_dict(torch.load('model.pth'))
model.eval()
print(model(x))

