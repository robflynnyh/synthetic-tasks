import torch, torch.nn as nn
from einops import rearrange

class RotaryPositionalEmbedding(torch.nn.Module): # TODO: incl fused kernel versions of rotary pos emb
    def __init__(
            self, 
            dim, 
            base=100000, 
            learned_freq=False,
            rotary_interpolation_factor=1.0,
            precision=torch.bfloat16, 
        ):
        """Rotary positional embedding
        Reference : https://blog.eleuther.ai/rotary-embeddings/
        Paper: https://arxiv.org/pdf/2104.09864.pdf
        Adapted from: https://fairseq.readthedocs.io/en/latest/_modules/fairseq/modules/rotary_positional_embedding.html
        Args:
            dim: Dimension of embedding
            base: Base value for exponential
            precision: precision to use for numerical values
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.learned_freq = learned_freq
        self.dim = dim

        if self.learned_freq:
            self.inv_freq = torch.nn.Parameter(inv_freq, requires_grad=True)
        else:
            self.register_buffer("inv_freq", inv_freq)

        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision # register rotary interpolation factor as buffer so it can be saved
        self.register_buffer("rotary_interpolation_factor", torch.tensor(rotary_interpolation_factor))

    def reset_if_needed(self):
        if self.learned_freq: # bcos we cant keep them after backward pass
            self.cos_cached = None
            self.sin_cached = None
            self.seq_len_cached = None
    
    def forward(self, seq_len, device=torch.device("cpu")):
        """
        Args:
            x: Input x with T X B X C
            seq_len: Sequence length of input x
        """
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq) / self.rotary_interpolation_factor
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers: (this should all just be moved into rotary class tbh)
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in earlier torch versions


def apply_rotary_pos_emb(q, k, cos, sin, q_offset: int = 0):
    q_cos, q_sin = (
        cos[:, q_offset : q.shape[1] + q_offset],
        sin[:, q_offset : q.shape[1] + q_offset],
    )
    return (q * q_cos) + (rotate_half(q) * q_sin), (k * cos) + (rotate_half(k) * sin)

class apply_rotary(): 
    def __init__(self, cos, sin, q_offset: int = 0, learned: bool = False):
        self.learned = learned
        self.cos = cos
        self.sin = sin
        self.q_offset = q_offset
    
    def __call__(self, q, k): return self.apply(q, k)
    def apply(self, q, k):
        q, k = map(lambda t: rearrange(t, 'b h n d -> b n h d'), (q, k))
        q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin, self.q_offset)
        return map(lambda t: rearrange(t, 'b n h d -> b h n d'), (q, k))

class RMSNorm(nn.Module): #https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

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