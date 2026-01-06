import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

class RotationalAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)

        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(T, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn = RotationalAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class SLM(nn.Module):
    def __init__(self, vocab_size, dim=512, layers=6, heads=8):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.lm_head(x)
