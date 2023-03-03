import torch
import torch.nn as nn
from torch.nn import functional as F

import math

import numpy as np

class Attention(nn.Module):
    def __init__(self, n_embd, n_head, bias, dropout):
        super().__init__()
        assert n_embd % n_head == 0

        self.key = nn.Linear(n_embd, n_embd, bias=bias)
        self.query = nn.Linear(n_embd, n_embd, bias=bias)
        self.value = nn.Linear(n_embd, n_embd, bias=bias)
        #self.proj = nn.Linear(n_embd, n_embd, bias=bias), don't have this here, have it in block building

        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

    def forward(self, x):
        pass


class MLP(nn.Module):
    def __init__(self, n_embd, bias, dropout, expansion, act):
        super().__init__()
        self.l1 = nn.Linear(n_embd, expansion * n_embd, bias=bias)
        self.l2 = nn.Linear(expansion * n_embd, n_embd, bias=bias)
        self.n_embd = n_embd
        self.act = act
        self.dropout = dropout

    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, norm, attn, mlp):
        super().__init__()
        self.norm = norm
        self.attn_to_mlp = nn.Linear(attn.n_embd, mlp.embd)
        self.attn = attn
        self.mlp = mlp

    def forward(self, x):
        #use attn_to_mlp to get dimensions correct
        #use self.norm to see if norm is needed
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class Model(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, dropout, blocks):
        super().__init__()

        assert n_embd == blocks[0].n_embd

        self.transformer = nn.ModuleDict(dict(
            w2v = nn.Embedding(vocab_size, n_embd),
            pos = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            blocks = blocks, #blocks will be a nn.ModuleList
        ))
        self.output_head = nn.Linear(n_embd, vocab_size, bias=False)
    
    def forward(self, idx, targets):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.w2v(idx) 
        pos_emb = self.transformer.pos(pos) 
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.blocks:
            x = block(x)

        logits = self.output_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)))

        return logits, loss

