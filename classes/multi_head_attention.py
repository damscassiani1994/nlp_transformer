import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from classes.rotary_positional_embedding import RotaryPositionalEmbedding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
       

        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)
    
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.shape
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, Q, K=None, V=None, mask=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        
        B, len_q, _ = Q.shape
        len_k = K.size(1)

        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        Q = self.rope(Q)
        K = self.rope(K)

        if mask is not None:
            while mask.dim() > 4:
                mask = mask.squeeze(0)
            if mask.dim() == 4:
                mask = mask.expand(B, self.num_heads, len_q, len_k)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1).expand(B, self.num_heads, len_q, len_k)
            
        attention_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, 
                                                          dropout_p=self.dropout.p if self.training else 0.0)
        attention_output = self.combine_heads(attention_output)

        ouput = self.W_O(attention_output)
        return ouput
    