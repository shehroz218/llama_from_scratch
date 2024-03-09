import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[int] = None
    norm_eps:float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None
    # device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0

    theta_numerator = torch.arange(0, head_dim, 2).float()
    # shape (head_dim /2)
    theta = 1.0 / (theta ** (theta_numerator/head_dim)).to(device)
    m = torch.arange(seq_len, device=device)
    # multiply each theta by each position using the outer product
    # shape: seqlen outer prodcut head_dim / 2 -> (seq_len, head_dim /2)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotarty_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, Seq_len, H, Head_dim) -> (B, Seq_len, H, Head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1 ,2))
    # seq_len, Head_dim/2 -> (1, Seq_len, 1, Head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, Seq_len, H, Head_dim /2) * (1, Seq_len, 1, Head_dim / 2) = (B, Seq_len, H, Head_dim / 2)
    x_rotated = x_complex * freqs_complex
    # (B,seq_len, H, Head_dim/2) -> (B, Seq_len, H, Head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, H, Head_dim/2, 2 ) -> (B, sseq_len, H, Head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x:torch.Tensor, n_rep:int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        x.unsqueeze(-2).expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        x.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-6):
        super().__init__()
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x:torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
    
    def forward(self, x:torch.Tensor):
        return self.weight * self.norm(x.float()).type_as(x)

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads

        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads*self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)

    def forward(self, x:torch.Tensor, start_pos:int, freqs_complex:torch.Tensor):
        batch_size, seq_len, _ = x.shape   #(B, 1, Dim)

        # (B, 1, Dim) -> (B, 1, n_heads* head_dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, n_kv_heads* head_dim)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # (B, 1, n_heads* head_dim) -> (B, 1, n_heads, head_dim)
        xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        #apply rotatory embeddings to the keys and values
        xq = apply_rotarty_embeddings(xq, freqs_complex, self.device)
        xk = apply_rotarty_embeddings(xk, freqs_complex, self.device)

        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

        #Retreiving the key value cvaches
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        valeus = self.cache_v[:batch_size, 0:start_pos+seq_len]

        #repeat the number of heads of K and V to reach the number of heads of Q
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1,2)
        keys = xk.transpose(1,2)
        values = xv.transpose(1,2)

        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, values)
        output = output.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = args.dim * 4
        hidden_dim = int(2 * hidden_dim / 3)

        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)


    def forward(self, x:torch.Tensor):
        return self.w2(self.w3(x) * F.silu(self.w1(x)))
        # swish = x * torch.sigmoid(x)
        # return self.w2(F.relu(self.w1(x))) + self.w3(swish)

    
    def forward(self, x:torch.Tensor):
        return self.fc2(F.relu(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(args.dim, eps =args.norm_eps)

        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x:torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1 

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList([])
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size , bias=False)


        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len *2, device = self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, seq_len)

        batch_size, seq_len = tokens.shape
        assert seq_len == 1

        # (B, Seq_len) -> (B, Seq_len, Dim)
        h = self.tok_embeddings(tokens)


        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len, :]

        #Consecutively apply all the encoder lauers
        for layer in self.layers:
            h = layer(h, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()

        return output
    