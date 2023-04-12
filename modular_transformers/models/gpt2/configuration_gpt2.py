from modular_transformers.models import components1
import torch.nn as nn
import transformers

"""
n_embd (`int`, *optional*, defaults to 768):
        Dimensionality of the embeddings and hidden states.
n_layer (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
n_head (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
n_inner (`int`, *optional*, defaults to None):
        Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
activation_function (`str`, *optional*, defaults to `"gelu"`):
        Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
"""
class GPT2Config(transformers.GPT2Config):
    def __init__(self, config):
        super().__init__(config)
        self.n_embd = [768]
        self.n_layer = len(self.n_embd)
        self.n_head = [12]
        self.n_inner = [3072]
        self.activation_function = [nn.GELU()]



# class GPT2Config():
#     """
#     Example:
#     ```python
#     >>> # Initializing a GPT2 configuration
#     >>> configuration = GPT2Config()
#     >>> # Initializing a model (with random weights) from the configuration
#     >>> model = GPT2Model(configuration)
#     >>> # Accessing the model configuration
#     >>> configuration = model.config
#     ```"""

#     def __init__(
#         self,
#         vocab_size=50257,
#         n_embd=768,
#         drop=0.1,
#         n_ctx = 1024
#     ):
#         #universal parameters
#         self.vocab_size = vocab_size
#         self.n_embd = n_embd
#         self.dropout = drop
#         self.n_ctx = n_ctx

#         #per block parameters (universal for GPT2)
#         n_head = 12
#         activation = nn.GELU()
#         n_layer = 12
#         bias = True
#         n_inner = None #4x default
#         self.blocks = nn.ModuleList()

#         for _ in range(n_layer):
#             attn = components1.Attention(master_embd = n_embd, n_embd=n_embd, n_head=n_head, bias=bias, dropout=drop, n_ctx=n_ctx)
#             mlp = components1.MLP(master_embd = n_embd, n_inner=n_inner, bias=bias, dropout=drop, activation=activation)
#             block = components1.Block(attn, mlp)
#             self.blocks.append(block)

