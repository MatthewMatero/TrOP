import math
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr, zscore
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import sys


class PositionalEncoding(nn.Module):
    """
    Sinusoidal pos encoding for transformers

    Args:
        dropout(float): dropout rate
        dim(int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pos_embed = torch.zeros(max_len, dim)
        position = torch.arange(0,max_len).unsqueeze(1)

        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pos_embed[:, 0::2] = torch.sin(position.float() * div_term)
        if dim % 2 != 0: # account for odd feature dimenisons
            div_term = div_term[:-1]
        pos_embed[:, 1::2] = torch.cos(position.float() * div_term)
        pos_embed = pos_embed.unsqueeze(0)
        
        self.register_buffer('pos_embed', pos_embed)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step:
            emb = emb + self.pos_embed[:, step][:,None,:]
        else:
            emb = emb + self.pos_embed[:, :emb.size(1)]
        
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pos_embed[:, :emb.size(1)]

class PositionalFeedForward(nn.Module):
    """
    FeedForward NN as defined in Vaswani 2017

    Args:
        input_dim(int): size of input entering the NN
        ff_dim(int): size of hidden layers of NN
        dropout(float): dropout rate
    """

    def __init__(self, input_dim, ff_dim, dropout=0.1):
        super(PositionalFeedForward, self).__init__()
        self.layer1 = nn.Linear(input_dim, ff_dim)
        self.layer2 = nn.Linear(ff_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)

    def forward(self, inputs):
        layer1_out = self.drop1(self.relu(self.layer1(self.layer_norm(inputs))))
        layer2_out = self.drop2(self.layer2(layer1_out))
        residual = layer2_out + inputs
        return residual
