#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:08:58 2020

@author: yuxing
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Hyper-parameters 
input_size = 28 * 28    # 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


# =============================================================================
# Logistic regression model
# =============================================================================

model = nn.Linear(input_size, num_classes)

# loss and optimizer

crit = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr = learning_rate)


total_step = len(train_loader)

for epoch in range(num_epochs):
    
    for i, (images, labels) in enumerate(train_loader):
        
        # Reshape the input into (batch_size, input_size)
        
        images = images.reshape(-1, input_size)
        
        # forward pass
    
        pred = model(images)
        loss = crit(pred, labels)
        
        # Backward and optimize
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        
# =============================================================================
# Test the model
# =============================================================================
        
# in the test phase, we don't need to compute the gradients

with torch.no_grad():
    
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
        
    
    


