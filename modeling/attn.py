import math
import torch
import torch.nn as nn
import sys

class MultiHeadedAttn(nn.Module):
    """
    Multi-Headed Attn

    Args:
        num_heads (int): amount of parallel heads
        model_dim (int): dimension of K,V,Q (must be divisible by head count)
        dropout (float): dropout rate
    """

    def __init__(self, num_heads, model_dim, dropout=0.1, use_final_linear=True):
        super(MultiHeadedAttn, self).__init__()
        
        assert model_dim % num_heads == 0
        self.dim_per_head = model_dim // num_heads
        self.model_dim = model_dim
        self.num_heads = num_heads

        self.linear_keys = nn.Linear(model_dim, num_heads*self.dim_per_head)
        self.linear_vals = nn.Linear(model_dim, num_heads*self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, num_heads*self.dim_per_head)

        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(p=dropout)
        self.use_final_linear = use_final_linear
        if self.use_final_linear:
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, val, query, mask=None, layer_cache=None, dtype=None):
        """
        Compute attention over context

        Args:
            key (FloatTensor): set of key_len key vectors ([batch, key_len, dim])
            value (FloatTensor): set of key_len value vectors ([batch, key_len, dim])
            query (FloatTensor): set of key_len query vectors ([batch, key_len, dim])
            mask (BooleanTensor): binary mask indicating which keys have non-zero attn ([batch, query_len, key_len])

        Returns:
            (FloatTensor, FloatTensor):

            * output context vectors [batch, query_len, dim]
            * one of the attn vectors [batch, query_len, key_len]
        """

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """ projection """
            return x.view(batch_size, -1, num_heads, dim_per_head).transpose(1,2)
        
        def unshape(x):
            """ compute context """
            return x.transpose(1,2).contiguous().view(batch_size, -1, num_heads*dim_per_head)
        
        if layer_cache is not None:
            if dtype == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif dtype == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key),\
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"],\
                                   layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            val = self.linear_vals(val)
            query = self.linear_query(query)
                
            key = shape(key)
            val = shape(val)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # calculate and scale scores
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2,3)) 

        if mask is not None:
            mask = ~mask # mask fill replaces where mask == 1
            mask = mask.unsqueeze(1).expand_as(scores)
            mask = mask.to(scores.get_device())
            scores = scores.masked_fill(mask, -1e18)
        
        # apply attn dropout and compute vectors
        attn = self.softmax(scores)
        drop_attn = self.drop(attn)
        
        if self.use_final_linear:
            context = unshape(torch.matmul(drop_attn, val))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, val)
            return context

