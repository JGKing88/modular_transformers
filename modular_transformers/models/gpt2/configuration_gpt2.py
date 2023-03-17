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
        n_positions=1024,
        n_embd=768,
        n_inner=None,
        activation_function=nn.GELU(),
        drop=0.1,
        block_size = 528
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.dropout = drop
        self.block_size = block_size

        attn1 = components.Attention(master_embd = n_embd, n_embd=480, n_head=12, bias=True, dropout=0.1, block_size=block_size)
        mlp1 = components.MLP(master_embd = n_embd, n_embd=300, bias=True, dropout=0.1, activation=nn.GELU())

        attn2 = components.Attention(master_embd = n_embd, n_embd=480, n_head=12, bias=True, dropout=0.1, block_size=block_size)
        mlp2 = components.MLP(master_embd = n_embd, n_embd=300, bias=True, dropout=0.1, activation=nn.GELU())

        attn3 = components.Attention(master_embd = n_embd, n_embd=768, n_head=12, bias=True, dropout=0.1, block_size=block_size)
        mlp3 = components.MLP(master_embd = n_embd, n_embd=768, bias=True, dropout=0.1, activation=nn.GELU())

        self.blocks = nn.ModuleList([components.Block(attn1, mlp1), components.Block(attn2, mlp2)])

