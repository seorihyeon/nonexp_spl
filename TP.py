import torch
import torch.nn as nn

class ANN(torch.nn.Module):
    def __init__(self, input_size, nodes, activation):
        super().__init__()
        layers = [torch.nn.Sequential(torch.nn.Linear(in_f, out_f), activation) for in_f, out_f in zip([input_size] + nodes, nodes)]
        layers.append(torch.nn.Linear(nodes[-1], 1))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, hist = False):
        lbl = []
        for layer in self.layers:
            lbl.append(x)
            x = layer(x)
        if hist:
            return x, lbl
        else:
            return x

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                           batch_first = True)
        
        self.fc = nn.Linear(in_features = hidden_size, out_features = 1)
        
    def forward(self, x):
       
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[0]).flatten()
        
        return out
        
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size = input_size,
                          hidden_size = hidden_size,
                          num_layers = num_layers,
                          batch_first = True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        _, hn = self.rnn(x)
        out = self.fc(hn[0]).flatten()
        return out

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size = input_size,
                          hidden_size = hidden_size,
                          num_layers = num_layers,
                          batch_first = True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        _, hn = self.gru(x)
        out = self.fc(hn[0]).flatten()
        return out