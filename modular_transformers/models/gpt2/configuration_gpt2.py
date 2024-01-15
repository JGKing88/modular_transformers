from modular_transformers.models import components
import torch.nn as nn
import transformers
from modular_transformers.models.loss_utils import l2_reg


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
    def __init__(self, config=None):
        
        if config == None:
            #train level configs are not being set, not sure if this is an issue
            super().__init__()
        else:
            if 'regsize' in config:
                regsize = config["regsize"]
            else: 
                regsize = 768
        
            super().__init__(hidden_size=regsize, vocab_size=config["vocab_size"], n_ctx=config["n_ctx"],bos_token_id=config["bos_token_id"],eos_token_id=config["eos_token_id"])
        
            self.bottleneck = config["bottleneck"]
            self.n_layer = config["n_layer"]
            if self.n_layer % 2 == 0:
                self.n_embds = [regsize] * int(self.n_layer/2 - 1) + [self.bottleneck, self.bottleneck] + [regsize] * int(self.n_layer/2 - 1)
            else:
                self.n_embds = [regsize] * int(self.n_layer//2) + [self.bottleneck] + [regsize] * int(self.n_layer//2)
                            
            self.n_heads = [4]*len(self.n_embds)
            self.n_inners = [4*x for x in self.n_embds]
            # self.n_inners = [768*4, 768*4, 128*4, 768*4, 768*4]
            self.activation_functions = ["gelu"]*len(self.n_embds)
            assert len(self.n_embds) == len(self.n_heads) == len(self.n_inners) == len(self.activation_functions)
                
            self.n_embd = self.n_embds[0]
            self.n_head = self.n_heads[0]
            self.n_inner = self.n_inners[0]
            self.activation_function = self.activation_functions[0]
                
            #0-indexed
            # self.loss_hooks = {1:l2_reg}
            self.loss_hooks = {}
                
            #make sure the loss hooks are valid
            for layer in self.loss_hooks.keys():
                assert layer < self.n_layer

    def get(self):
        #return as dictionary
        return {
                "n_embds":self.n_embds,
                "n_heads":self.n_heads,
                "n_inners":self.n_inners,
                "activation_functions":self.activation_functions,
                "n_layer":self.n_layer,
                "n_embd":self.n_embd,
                "n_head":self.n_head,
                "n_inner":self.n_inner,
                "activation_function":self.activation_function,
                "loss_hooks":self.loss_hooks,
                "bottleneck":self.bottleneck
        }


# class GPT2miniConfig(PretrainedConfig):
#     def __init__(self, config):
        
#         self.n_embds = [8] #list of embedding size for the rest of the blocks
#         self.n_heads = [4]
#         self.n_inners = [32]
#         self.activation_functions = ["gelu"]

#         self.n_layer = len(self.n_embds)
#         self.n_embd = self.n_embds[0]
#         self.n_head = self.n_heads[0]
#         self.n_inner = self.n_inners[0]
#         self.activation_function = self.activation_functions[0]
#         super(GPT2miniConfig,self).__init__(vocab_size=config["vocab_size"], n_ctx=config["n_ctx"], bos_token_id=config["bos_token_id"],
#                          eos_token_id=config["eos_token_id"])


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

