#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:50:08 2020

@author: yuxing
"""

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


torch.random.manual_seed(23)


# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)



model = nn.Linear(input_size, output_size)

crit = nn.MSELoss()

opt = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    
    # forward pass
    pred = model(inputs)
    loss = crit(pred,targets)
    
    # backward and optimize
    
    opt.zero_grad()
    loss.backward()
    opt.step()

    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    
model(torch.from_numpy(x_train)).data.numpy()

predicted = model(torch.from_numpy(x_train)).data.numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()


# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
