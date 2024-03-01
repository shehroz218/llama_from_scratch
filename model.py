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
    otefloat = 1e-5

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

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-6):
        super().__init__()
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x:torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
    
    def forward(self, x:torch.Tensor):
        return self.weight * self.norm(x.float()).type_as(x)




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
    