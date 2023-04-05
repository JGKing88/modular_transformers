from modular_transformers.models import components
import torch.nn as nn

class GPT2Config():
    """
    Example:
    ```python
    >>> # Initializing a GPT2 configuration
    >>> configuration = GPT2Config()
    >>> # Initializing a model (with random weights) from the configuration
    >>> model = GPT2Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        vocab_size=50257,
        n_embd=768,
        drop=0.1,
        n_ctx = 1024
    ):
        #universal parameters
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.dropout = drop
        self.n_ctx = n_ctx

        #per block parameters (universal for GPT2)
        n_head = 12
        activation = nn.GELU()
        n_layer = 12
        bias = True
        n_inner = None #4x default
        self.blocks = nn.ModuleList()

        for _ in range(n_layer):
            attn = components.Attention(master_embd = n_embd, n_embd=n_embd, n_head=n_head, bias=bias, dropout=drop, n_ctx=n_ctx)
            mlp = components.MLP(master_embd = n_embd, n_inner=n_inner, bias=bias, dropout=drop, activation=activation)
            block = components.Block(attn, mlp)
            self.blocks.append(block)

