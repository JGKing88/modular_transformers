import transformers
import torch
import torch.nn as nn 
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP
from transformers.activations import ACT2FN
from transformers.pytorch_utils import Conv1D

from typing import Optional, Tuple, Union

class Transformer(transformers.GPT2Model):
    def __init__(self, config=None):
        super().__init__(config)
        self.h = nn.ModuleList()
        self.hook_functions = {}
        self.hook_losses = {}

        for layer in range(config.n_layer):
            #each block gets a new GPT2Config object
            #each block needs an output dimension to be given to 
            #it's dimension changer final layers
            #the last block needs to return the embedding dimension to the original size
            if layer == config.n_layer - 1:
                output_dim = config.n_embd
            else:
                output_dim = config.n_embds[layer+1]
            
            block_config = transformers.GPT2Config(n_embd = config.n_embds[layer], 
                                                   hidden_size = config.n_embds[layer],
                                                   n_head = config.n_heads[layer], 
                                                   n_inner = config.n_inners[layer], 
                                                   activation_function = config.activation_functions[layer],
                                                   output_dim = output_dim,)
            block = TransformerBlock(block_config)
            self.h.append(block)
            
            if layer in self.config.loss_hooks:
                #very important to figure about what part of attention block is desired
                #currently using the input of c_proj
                attention = block.attn.c_proj
                attention.register_forward_hook(self.record_extra_loss)
                #all you get in a hook is the module, inputs, and outputs. So the only way to link a module
                #to a loss function is to use a dictionary
                self.hook_functions[attention] = self.config.loss_hooks[layer]
                self.hook_losses[attention] = 0
        
    def record_extra_loss(self, module, inputs, output):
        loss = self.hook_functions[module](inputs)
        self.hook_losses[module] += loss
        #if you want to backpropagate the loss from here, uncomment the following line
        #loss.backward()

    def output_extra_losses(self):
        intm_dict = self.hook_losses.copy() #so that we can clear the dict
        for key in self.hook_losses.keys():
            self.hook_losses[key] = 0
        return intm_dict

"""
An exact replica of the hugging face GPT2Block except of the final embedding size
shifter layer
"""
class TransformerBlock(nn.Module):
    def __init__(self, config=None, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)
        
        #MY ADDITION, NEED TO CHANGE THE EMBEDDING SIZE
        self.dim_changer = nn.Linear(hidden_size, config.output_dim, bias=False)
        for param in self.dim_changer.parameters():
            param.requires_grad = False

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        hidden_states = self.dim_changer(hidden_states) #MY ADDITION

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
    


class LM(transformers.GPT2LMHeadModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.transformer = Transformer(config)

    def output_extra_losses(self):
        return self.transformer.output_extra_losses()