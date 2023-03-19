import math
import torch
import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from Model import Model
from myLinear import myLinear as linear
from torch.autograd import Variable
from torch.nn import functional as F, Parameter
from se import SELayer
from numpy.random import RandomState

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

class ConvD(Model):

    def __init__(self, config):
        super(ConvD, self).__init__(config)
        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_embeddings_r = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.perm = self.config.p_norm
        self.range = self.reRange()
        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv_layer = nn.Conv2d(1, self.config.out_channels, (2, self.config.kernel_size))  # kernel size x 3
        # self.se = SELayer(self.config.out_channels, self.config.hidden_size)
        # self.chequer_perm = torch.LongTensor(np.int32([np.random.permutation(self.config.hidden_size)]))
        self.conv2_bn = nn.BatchNorm2d(self.config.out_channels)
        self.conv3_bn = nn.BatchNorm1d(self.config.hidden_size)
        self.map_dropout = nn.Dropout2d(self.config.map_drop)
        self.dropout = nn.Dropout(self.config.convkb_drop_prob)
        self.non_linearity = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(self.config.negative_slope)
        self.p_layer = linear(self.config.hidden_size, self.config.out_channels)
        self.fc_layer_1 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc_layer = nn.Linear((self.config.hidden_size + 1 - self.config.kernel_size), 1, bias=False)
        self.criterion = nn.Softplus()
        self.init_parameters()

    def init_parameters(self):
        if self.config.use_init_embeddings == False:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            nn.init.xavier_normal_(self.rel_embeddings_r.weight.data)
        else:
            self.ent_embeddings.weight.data = self.config.init_ent_embs
            self.rel_embeddings.weight.data = self.config.init_rel_embs
            
    def reRange(self):
        seq = np.int32(np.random.permutation(self.config.hidden_size))
        temp = np.int32(np.arange(0, self.config.hidden_size * 2, 1))
        for i in range(self.config.hidden_size):
            if seq[i] % self.perm == 0:
                temp[i] = i + self.config.hidden_size
                temp[i+self.config.hidden_size] = i 
        chequer_perm = torch.LongTensor(temp)
        return chequer_perm

    def _calc(self, h, t):
        conv_input = torch.cat([h, t], 1) 

        conv_input = conv_input.view(-1, 1, 2, self.config.hidden_size)
        conv_input = self.conv1_bn(conv_input)
        out_conv = self.conv_layer(conv_input)
        out_conv = self.conv2_bn(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = self.map_dropout(out_conv)
        out_conv = out_conv.view(-1, self.config.out_channels, self.config.hidden_size)
        out_conv = out_conv.transpose(1, 2)
        out_conv = self.p_layer(out_conv)
        out_conv =self.leaky_relu(out_conv)
        out_conv = self.conv3_bn(out_conv)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc).view(-1)
        return -score

    def loss(self, score, regul):
        return torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul

    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        r_h = self.rel_embeddings(self.batch_r)
        t = self.ent_embeddings(self.batch_t)
        r_t = self.rel_embeddings_r(self.batch_r)
        h = h * r_h
        t = t * r_t
        score = self._calc(h, t)

        # regularization
        l2_reg = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r_h ** 2) + torch.mean(r_t ** 2)
        for W in self.conv_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.fc_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.p_layer.parameters():
            l2_reg = l2_reg + W.norm(2)


        return self.loss(score, l2_reg)

    def predict(self):

        h = self.ent_embeddings(self.batch_h)
        r_h = self.rel_embeddings(self.batch_r)
        t = self.ent_embeddings(self.batch_t)
        r_t = self.rel_embeddings_r(self.batch_r)
        h = h * r_h
        t = t * r_t
        score = self._calc(h, t)

        return score.cpu().data.numpy()
