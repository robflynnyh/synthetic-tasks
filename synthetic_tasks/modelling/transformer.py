import torch, torch.nn as nn
from einops import rearrange
from synthetic_tasks.modelling.pos_enc import RotaryPositionalEmbedding, apply_rotary
from synthetic_tasks.modelling.norm import RMSNorm

class Attention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model, self.heads = d_model, heads

        self.qkv = nn.Linear(d_model, 3*d_model)
        self.fout = nn.Linear(d_model, d_model)

    def forward(self, x, mask, pos_fn=None):
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        if pos_fn is not None: q, k = pos_fn(q, k)
        a_out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        a_out = rearrange(a_out, 'b h n d -> b n (h d)')
        return self.fout(a_out)

class SwiGlu(nn.Module):
    def __init__(self, d_model, exp_f=2):
        super().__init__()
        self.d_model, self.exp_f = d_model, exp_f

        self.fc1 = nn.Linear(d_model, d_model * exp_f*2)
        self.fc2 = nn.Linear(d_model * exp_f, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(self.act(a) * b)
    
class PreNorm(nn.Module):
    def __init__(self, d_model, fn):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)



class Model(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            d_model, 
            heads,
            n_layers,
    ):
        super().__init__()
        self.vocab_size, self.d_model, self.heads, self.n_layers = vocab_size, d_model, heads, n_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.out = PreNorm(d_model, nn.Linear(d_model, 1))
     
        self.pe = RotaryPositionalEmbedding(d_model // heads)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(d_model, SwiGlu(d_model))),
                Residual(PreNorm(d_model, Attention(d_model, heads))),
            ]))
    
    def total_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, mask=None):
        x = self.embedding(x)
        pos_fn = apply_rotary(*self.pe(x.shape[1], device=x.device))

        for i, (swiglu, attn) in enumerate(self.layers): x = attn(swiglu(x), mask, pos_fn)

        return self.out(x)  