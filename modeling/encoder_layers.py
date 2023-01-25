"""
    Defines the individual layers to be used for a transformer encoder
"""

import math
import torch.nn as nn
import torch

from modeling.attn import MultiHeadedAttn
from modeling.neural import PositionalEncoding, PositionalFeedForward

class TransformerEncoderLayer(nn.Module):
    """
    A single transformer block for encoding

    Args:
        model_dim (int): dimension of key/val/query in MHAttn,
                         Also input size of first layer for positonal FF
        nheads (int): number of heads for MHattn
        ff_dim (int): dimension of positional FF
        dropout (float): dropout rate
    """

    def __init__(self, model_dim, nheads, ff_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttn(nheads, model_dim, dropout=dropout)
        self.feed_forward = PositionalFeedForward(model_dim, ff_dim, dropout)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, inputs, mask):
        """
        Transformer forward pass

        Args:
            inputs (FloatTensor): [batch_size, src_len, model_dim]
            mask (LongTensor): [batch_size, src_len]
        Returns:
            (FloatTensor):

            * outputs [batch_size, src_len, model_dim]
        """
        input_norm = self.layer_norm(inputs)
        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.drop(context) + inputs
        return self.feed_forward(out)
