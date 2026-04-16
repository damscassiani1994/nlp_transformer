import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        in_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        pos = torch.arange(max_seq_len).float()
        angles = pos[:, None] * in_freq[None, :]

        sin = torch.sin(angles)
        cos = torch.cos(angles)

        self.register_buffer('sin', sin)
        self.register_buffer('cos', cos)

    def rotate_half(self, x, cos=None, sin=None):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rot = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        return rot.flatten(-2)
    
    def forward(self, x):
        # x shape: (batch_size, num_heads, seq_len, d_k)
        batch_size, num_heads, seq_len, d_k = x.shape
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)

        sin = sin.to(x.device)
        cos = cos.to(x.device)

        return self.rotate_half(x, cos=cos, sin=sin)
