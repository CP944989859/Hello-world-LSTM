__author__ = 'CP'
import torch
from torch.autograd import Variable
class LSTMcell(torch.nn.Module):
    def __init__(self, input_size, num_hidden, x2h_bias = False, h2h_bias = False):
        super(LSTMcell, self).__init__()
        self.num_hidden = num_hidden
        self.input_size = input_size
        self.x2h = torch.nn.Linear(input_size, num_hidden, bias = x2h_bias)
        self.h2h = torch.nn.Linear(num_hidden, num_hidden, bias = h2h_bias)

    def forward(self, input, hx):
        x = input
        # print(x.shape, hx.shape)
        gates = self.x2h(x) + self.h2h(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4,1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = torch.mul(x, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.tanh(cy))

        # return (hy,cy)
        return hy,cy

class LSTM(torch.nn.Module):
    def __init__(self,cell_class, num_layer, input_size, hidden_size, use_bias = False):
    # def __init__(self, LSTMcell, num_layer, input_size, hidden_size, use_bias = False):
        super(LSTM,self).__init__()
        self.num_layers = num_layer
        self.hidden_size = hidden_size
        for layer in range(num_layer):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(layer_input_size, hidden_size, use_bias)
            # cell = LSTMcell(layer_input_size, hidden_size, use_bias)
            setattr(self, 'cell_{}'.format(layer), cell)

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_param(self):
        for layer in range(self.num_layer):
            cell = self.get_cell(layer)
            cell.reset_param()

    def forward(self, input_, hx = None):
        max_time, batch_size, _ = input_.size()
        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())   ##
            # hx = [(hx, hx) for _ in range(self.num_layers)]
            hx = [(hx) for _ in range(self.num_layers)]

        layer_output = None
        new_hx = []
        for layer in range(self.num_layers):
            print("layer:   ", layer)
            cell = self.get_cell(layer)
            print(cell)
            layer_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(cell = cell, input_= input_, hx = hx[layer])
            input_ = layer_output
            new_hx.append((layer_h_n, layer_c_n))
        output = layer_output  ##
        return output, new_hx

    @staticmethod
    def _forward_rnn(cell, input_, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            print(time,'>>>>>>>>>>>',input_[time].shape,len(hx))
            # h_next, c_next = cell(input_[time], hx, time)
            h_next, c_next = cell(input_[time], hx)
            # print(input_[time])
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output,0)
        return output, hx












