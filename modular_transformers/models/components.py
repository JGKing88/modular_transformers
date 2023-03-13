import torch
from torch import nn
from torch.nn import functional as F
import math

class Attention(nn.Module):
    def __init__(self, master_embd, n_embd, n_head, bias, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0

        self.key = nn.Linear(master_embd, n_embd, bias=bias)
        self.query = nn.Linear(master_embd, n_embd, bias=bias)
        self.value = nn.Linear(master_embd, n_embd, bias=bias)
        self.head = nn.Linear(n_embd, master_embd, bias=bias)

        self.ln = nn.LayerNorm(master_embd)

        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        C = self.n_embd #reset C to n_embd

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # compute attention per head
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        x = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        x = x.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        x = self.head(x)
        x = self.ln(self.dropout(x))
        return x


class MLP(nn.Module):
    def __init__(self, master_embd, n_embd, bias, dropout, activation):
        super().__init__()
        #TODO: figure out if master_embd is a worthwhile addition
        self.l1 = nn.Linear(master_embd, n_embd, bias=bias)
        self.l2 = nn.Linear(n_embd, master_embd, bias=bias)
        self.ln = nn.LayerNorm(master_embd)
        self.n_embd = n_embd
        self.act = activation
        self.dropout = nn.Dropout(dropout)
        self.master_embd = master_embd

    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        x = self.act(x)
        x = self.ln(self.dropout(x))
        return x


class Block(nn.Module):
    def __init__(self, attn, mlp):
        super().__init__()
        self.attn = attn
        self.mlp = mlp

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x
    
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        n_embd = config.n_embd
        vocab_size = config.vocab_size
        dropout = config.dropout
        blocks = config.blocks
        block_size = config.block_size

        assert block_size == blocks[0].attn.block_size, "block size must be the same for all blocks"

        self.transformer = nn.ModuleDict(
            dict(
                w2v=nn.Embedding(vocab_size, n_embd),
                pos=nn.Embedding(block_size, n_embd),
                drop=nn.Dropout(dropout),
                blocks=blocks,  # blocks will be a nn.ModuleList
                ln_f = nn.LayerNorm(n_embd)
            )
        )
        self.output_head = nn.Linear(n_embd, vocab_size, bias=False)

        #TODO: set up initialization scheme
            #i.e. modular initialization, could have init functions in a different util file
        self.apply(self._init_weights)

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        tok_emb = self.transformer.w2v(idx)
        pos_emb = self.transformer.pos(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.blocks:
            x = block(x)

        logits = self.output_head(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx