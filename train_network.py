import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import os
import math
import copy
import time
import sys
import matplotlib.pyplot as plt
from scipy.io import savemat

import utils
import TP

def train(model, optimizer, loss_fn, data, height = 10, recur = True, epochs = 100, batch_number = 256, schedule = True):
    print('train start')
    model.train()
    if schedule:
        op_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [math.floor(0.7*epochs), math.floor(0.8*epochs), math.floor(0.9*epochs), math.floor(0.98*epochs)], gamma = 0.1)
    mlen = 0; epoch_time_sum = 0; mean_epoch_time = 0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        data.shuffle_data()
        for r in range(math.ceil(data.train_data_len/batch_number)):
            X, Y = data.get_data(batch_number,0)
            if recur:
                X = utils.stack_data(X, height)
                
            X = torch.tensor(X, dtype = torch.float).to('cuda');
            if recur:
                Y = torch.tensor(Y, dtype = torch.float).flatten().to('cuda')
            else:
                Y = torch.tensor(Y, dtype = torch.float).to('cuda')
            
            Y_pred = model(X)
            loss = loss_fn(Y_pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del X; del Y;
            torch.cuda.empty_cache()
        epoch_end_time = time.time()
        epoch_time_consumed = epoch_end_time - epoch_start_time
        epoch_time_sum = epoch_time_sum + epoch_time_consumed
        if epoch % 10 == 1:
            mean_epoch_time = round(epoch_time_sum / (epoch+1), 4)
            print("\r" + ' '*mlen + "\r", end = '')
            loss_state = str(epoch) + "\t" + str(loss.item())
            et_state = "\tmet: " + str(mean_epoch_time)  + "\tert: " + str(mean_epoch_time*(epochs - epoch + 1))
            state = loss_state + et_state
            print(state, end = '')
            mlen = len(state)
        if schedule:
            op_scheduler.step()
    print('\ntrain end')

if __name__ == '__main__':
    structure = sys.arg[1]
    machine_number = sys.arg[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_data = utils.linedataset(utils.load_data_from_csv("data", machine_number, "train", mode = mode));
    test_data = utils.load_data_from_csv("data", machine_number, "test", mode = mode);
    
    sample_x, sample_y = train_data.get_data(1)
    input_size = sample_x.shape[1]
    node_num = 1000
    layer_number = 2
    path = "pt/" + str(machine_number) + "m/"
    result = np.zeros((10,3))
    h = 10
    for n in range(len(node_num)):
        if structure == "ANN":
            recur = False
            model = TP.ANN(input_size, [node_num[n]]*layer_number, torch.nn.ReLU())
        else:
            recur = True
            if structure == "RNN":
                model = TP.RNN(input_size, node_num[n], layer_number)
            elif structure == "LSTM":
                model = TP.LSTM(input_size, node_num[n], layer_number)
            elif structure == "GRU":
                model = TP.GRU(input_size, node_num[n], layer_number)
        model = model.to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
        loss_fn = torch.nn.HuberLoss(reduction = 'sum')
        train(model, optimizer, loss_fn, train_data, recur= recur)
        
        torch.save(model, path + structure + ".pt")
    del model, optimizer, loss_fn
    torch.cuda.empty_cache()