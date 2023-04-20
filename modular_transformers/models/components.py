import transformers
import torch
import torch.nn as nn 
from typing import Optional, Tuple, Union
from torch.cuda.amp import autocast


class LM(transformers.GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        #each block gets a new GPT2Config object
        self.h = nn.ModuleList()

        for layer in range(config.n_layer):
            block_config = transformers.GPT2Config(n_embd = config.n_embds[layer], n_head = config.n_heads[layer], n_inner = config.n_inners[layer], activation_function = config.activation_functions[layer])
            self.h.append(transformers.models.gpt2.modeling_gpt2.GPT2Block(block_config))
