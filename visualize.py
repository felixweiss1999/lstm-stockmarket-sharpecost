import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt('averageLosses.txt')
plt.plot(list(range(len(a)+1))[1:], a)
plt.xlabel('Epochs')
plt.ylabel('Sharpe Ratio')
plt.ylim([0,1])
plt.xticks([1,10,20,30,40,50,60,70,80,90,100])
plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.show()


b = np.loadtxt('losses.txt')
plt.plot(list(range(len(b[0]))), b[0], label='Epoch 0')
plt.plot(list(range(len(b[99]))), b[99], label='Epoch 99')
leg = plt.legend(loc='upper center')
plt.xticks(list(range(len(b[0]))))
plt.xlabel('Batch')
plt.ylabel('Average Sharpe Ratio')

plt.show()
