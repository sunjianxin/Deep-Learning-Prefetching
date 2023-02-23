import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first = True)
        # If your input data is of shape (seq_len, batch_size, features)
        # then you don’t need batch_first=True and your LSTM will give
        # output of shape (seq_len, batch_size, hidden_size).

        # If your input data is of shape (batch_size, seq_len, features)
        # then you need batch_first=True and your LSTM will give
        # output of shape (batch_size, seq_len, hidden_size).
        
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.hidden = (torch.zeros(self.num_layers, 1, hidden_size), torch.zeros(self.num_layers, 1, hidden_size))
        
    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim,
                                      x.size(0), # batch_size, retrieved from batch data x
                                      self.hidden_dim)).cuda()
        else:
            h0 = Variable(torch.zeros(self.layer_dim,
                                      x.size(0), # batch_size, retrieved from batch data x
                                      self.hidden_dim))
        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim,
                                      x.size(0), # batch_size, retrieved from batch data x
                                      self.hidden_dim)).cuda()
        else:
            c0 = Variable(torch.zeros(self.layer_dim,
                                      x.size(0), # batch_size, retrieved from batch data x
                                      self.hidden_dim))
        # print("x.size(0)", x.size(0))
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # x is (batch_size, seq_len, features)
        
        out = self.fc(out[:, -1, :])
        
        return out
