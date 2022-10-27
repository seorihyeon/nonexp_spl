import numpy as np
import json
import os
import random
import math
import copy
    
class linedataset():
    def __init__(self, data, validation_rate = 0.05):
        self.data = data
        self.idx = 0
        self.len = data["input"].shape[0]
        self.validation_data_len = math.ceil(self.len * validation_rate)
        self.train_data_len = self.len - self.validation_data_len;
        self.data_order = np.array(range(self.len))
        self.validation_index = random.sample(list(range(self.len)), self.validation_data_len)
        mask = np.ones(self.len, dtype = bool);
        mask[self.validation_index] = False;
        
        self.train_data_index = self.data_order[mask,...]
        self.validation_data_index = self.data_order[~mask,...]
        
    def reset_idx(self):
        self.idx = 0

    def shuffle_data(self):
        random.shuffle(self.data_order)
        
        self.validation_index = random.sample(list(range(self.len)), self.validation_data_len)
        mask = np.ones(self.len, dtype = bool);
        mask[self.validation_index] = False;
        
        self.train_data_index = self.data_order[mask,...]
        self.validation_data_index = self.data_order[~mask,...]
        
        self.reset_idx()

    def get_data(self, size, index = -1, validation = False):
        if validation:
            x_batch = self.data["input"][self.validation_data_index]
            y_batch = self.data["output"][self.validation_data_index]
        else:
            endidx = self.idx + size
            if endidx > self.train_data_len:
                endidx = self.train_data_len
            didx = self.train_data_index[self.idx : endidx]
            if endidx == self.train_data_len:
                self.reset_idx()
            else:
                self.idx = endidx
            x_batch = self.data["input"][didx]
            y_batch = self.data["output"][didx]
        if index != -1:
            y_batch = y_batch[:,index]
            y_batch = y_batch.reshape(y_batch.shape[0],1)
        return x_batch, y_batch
        
def weibull_mean(lamb, k):
    return lamb*math.gamma(1+(1/k))

def weibull_var(lamb, k):
    return pow(lamb,2)*(math.gamma(1+(2/k))-pow(math.gamma(1+(1/k)),2))

def load_data_from_csv(data, m_num, label, mode = 0):
    folder_path = data + "/" + str(m_num) + "m/data"
    result = {}
    file_path = os.path.join(folder_path, "xdata" + "_" + label + ".csv")
    _input = np.genfromtxt(file_path, delimiter = ",")
    if mode == 0:
        result["input"] = np.hstack((_input, 1/_input))
    elif mode == 1:
        result["input"] = _input
    
    file_path = os.path.join(folder_path, "ydata" + "_" + label + ".csv")
    result["output"] = np.genfromtxt(file_path, delimiter = ",")
        
    return result

def stack_data(data, height):
    dlen = data.shape[0]
    width = data.shape[1]
    stacked_data = np.zeros((dlen, height, width))
    for i in range(dlen):
        line = data[i,:]
        tmp = np.zeros((height, width))
        for j in range(height):
            tmp[j,:] = line
        stacked_data[i,:,:] = tmp
    return stacked_data

def get_parameter_from_json(file_name):
    with open(file_name, "r") as json_file:
        json_data = json.load(json_file)
        machine_number = json_data['M']
        X_input = json_data["m1"]["up"] + json_data["m1"]["down"]
        for i in range(2,machine_number+1):
            if machine_number == 2:
                X_input = X_input + [json_data["N"]]
            else:    
                X_input = X_input + [json_data["N"][i-2]]
            machine = "m" + str(i)
            X_input = X_input + json_data[machine]["up"] + json_data[machine]["down"]

    return np.array(X_input), machine_number