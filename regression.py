#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:34:27 2020

@author: yuxing
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


torch.manual_seed(1)

# create data
x = torch.unsqueeze(torch.linspace(-1,1,100), dim = 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

#print(x)

x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy(), y.data.numpy())




class Net(nn.Module):

    def __init__(self, n_features, n_hidden, n_output):
        
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_features, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)
        
        
    def forward(self, x):
 
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        
        return x


net = Net(1, 10, 1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)

loss_func = torch.nn.MSELoss()

for t in range(100):
    
    prediction = net(x)
    
    loss = loss_func(prediction, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
#    if t%5 == 0:
#        
#        # plot the learning process
#        
#        plt.cla()
#        plt.scatter(x.data.numpy(), y.data.numpy())
#        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw = 5)
#        plt.text(0.5, 0, 'loss = {}'.format(loss.item()))
#        plt.pause(0.1)
        
        

# =============================================================================
# save and restore the net
# =============================================================================


torch.manual_seed(1)    # reproducible

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

def save():
    
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.3)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    torch.save(net1, "net.pkl")   # save the entire net.
    
    torch.save(net1.state_dict(), 'net_params.pkl') # save the parameters
    
    
    prediction = net1(x)
    
    plt.subplot(131)
    plt.title("net1")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw = 5)


def restore_net():
    
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    
    plt.subplot(132)
    plt.title("net2")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw = 5)
    
    
def restore_params():
    
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)
    
    plt.subplot(133)
    plt.title("net3")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw = 5)
    
    
save()



restore_net()

restore_params()
