import os
import math
import time
import numpy as np
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TPnet(nn.Module):
    """A class for building RNN model"""
    
    def __init__(self, M, height = 10, hidden_size = 1000, num_layer = 2, activation = 'relu'):
        super(TPnet, self).__init__()
        
        input_size = 2*(5*M-1);
        self.height = 10
        
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = num_layer, nonlinearity = activation, batch_first = True)
        self.fc = nn.Linear(hidden_size, 1);
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x):
        x, _status = self.rnn(x)
        h_t = x[:,-1,:]
        return self.fc(h_t)

    def train(self, data, optimizer, loss_fn, epochs = 1000, batch_number = 256, schedule = True):
        self.train()
        if schedule:
            _milestones = [math.floor(0.7*epochs), math.floor(0.8*epochs), math.floor(0.9*epochs), math.floor(0.98*epochs)]
            op_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = _milestones, gamma = 0.1)
        
        for epoch in range(epochs):
            data.shuffle_data()
            for r in range(math.ceil(data.train_data_len/batch_number)):
                X, Y = data.get_data(batch_number, 1)
                X = torch.tensor(utils.stack_data(X, self.height), dtype = torch.flaot).to(self.device)
                Y = torch.tensor(Y, dtype = torch.float).to(self.device)
                
                Y_pred = self(X)
                loss = loss_fn(Y_pred, Y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del X; del Y;
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            if schedule:
                op_scheduler.step()
                
    def validate(self, data):
        self.eval()
        X, Y = data.get_data(0, 1, True)
        X = torch.tensor(utils.stack_data(X, self.height), dtype = torch.float).to(self.device)
        
        Y_pred = self(X).detach().to('cpu').numpy()
        
        num_data = Y_pred.shape
        errs = np.zeros(num_data)
        for idx in range(num_data[0]):
            for jdx in range(num_data[1]):
                true_value = Y[idx][jdx]
                pred_value = Y_pred[idx][jdx]
                errs[idx][jdx] = (abs(true_value - pred_value)/true_value)*100
        
        em = np.mean(errs,axis = 0)
        es = np.std(errs, axis = 0)
        
        return errs, em, es
    
    def test(self, data, batch_number = 1000):
        self.eval()
        
        X = data["input"]
        Y = data["output"][:,1]; Y = Y.reshape(Y.shape[0],1)
        X = torch.tensor(utils.stack_data(X, self.height), dtype = torch.float).to(self.device)
        
        first = True
        for r in range(math.ceil(X.shape[0]/batch_number)):
            if batch_number*(r+1) < X.shape[0]:
                X_batch = X[batch_number*r:batch_number*(r+1)][:]
            else:
                X_batch = X[batch_number*r:X.shape[0]][:]
            
            Y_batch = self(X_batch)
            
            if first:
                Y_pred = Y_batch
                first = False
            else:
                Y_pred = torch.vstack((Y_pred, Y_batch))
                
        Y_pred = Y_pred.detach().to('cpu').numpy()
        
        num_data = Y_pred.shape
        errs = np.zeros(num_data)
        for idx in range(num_data[0]):
            for jdx in range(num_data[1]):
                true_value = Y[idx][jdx]
                pred_value = Y_pred[idx][jdx]
                errs[idx][jdx] = (abs(true_value - pred_value)/true_value)*100
        
        em = np.mean(errs,axis = 0)
        es = np.std(errs, axis = 0)
        
        return errs, em, es
    
    def predict_TP(self, parameters):
        machine_number = (parameters.size+1)/5
        input = np.hstack((parameters, 1/parameters))
        input = input.reshape(1, input.shape[0])

        X = utils.stack_data(input, self.height)
        X = torch.tensor(X, dtype = torch.float);
        Y_pred = self(X)
        return Y_pred.detach().numpy()