import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn

class myLinear(nn.Module):

    def __init__(self, h_features, w_features, bias=True):
        super(myLinear, self).__init__()
        self.h_features = h_features
        self.w_features = w_features
        comp = Parameter(torch.zeros(1, w_features))
        comp_bias = Parameter(torch.zeros(1))
        for i in range(h_features):
            self.weight = Parameter(torch.Tensor(1, w_features))
            if bias:
                self.bias = Parameter(torch.zeros(w_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
            if i == 0:
                comp = self.weight
                if bias:
                    comp_bias = self.bias
            else:
                comp = torch.cat([comp, self.weight])
                if bias:
                    comp_bias = torch.cat([comp_bias, self.bias])
        print(comp.shape)
        self.weight = Parameter(comp)
        if bias:
            self.bias = Parameter(comp_bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)

    def forward(self, input):
        output = input * self.weight
        if self.bias is not None:
            bia = self.bias.view(1, self.h_features, self.w_features)
            output += bia
        ret = torch.sum(output, -1)
        return ret
