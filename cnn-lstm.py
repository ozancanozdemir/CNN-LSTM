# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:31:10 2022

@author: ozancan ozdemir
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(self, input_size, output_size,hidden_size,num_layers):
        super(CNNLSTM, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(64,32,kernel_size=1, stride = 1, padding=1)
        self.batch1 =nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32,32,kernel_size=1, stride = 1, padding=1)
        self.batch2 =nn.BatchNorm1d(32)
        self.LSTM = nn.LSTM(input_size=5, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(32*hidden_size, output_size)
        #self.fc2 = nn.Linear(1, 1)
        

    def forward(self, x):
        #in_size1 = x.size(0)  # one batch
        x = F.selu(self.conv1(x))
        x = self.conv2(x)
        x = F.selu(self.batch1(x))
        x = self.conv3(x)
        x = F.selu(self.batch2(x))
        x, h = self.LSTM(x) 
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
        #in_size1 = x.size(0)  # one batch
        #x = x.view(in_size1, -1)
        # flatten the tensor x[:, -1, :]
        x = self.fc1(x)
        output = torch.sigmoid(x)
        #output = self.fc2(x)

    
        
        return output


model = CNNLSTM(input_size, output_size,hidden_size, num_layers)
print(model)


num_epochs = 800
learning_rate = 0.01
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
loss_list = []
# Train the model
for epoch in range(num_epochs):
    outputs = model(trainX)
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = criterion(outputs, trainY)
    loss_list.append(loss)
    loss.backward()
    
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))