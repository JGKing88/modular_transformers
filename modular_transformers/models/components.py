import transformers
import torch
import torch.nn as nn 
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP
from transformers.activations import ACT2FN
from transformers.pytorch_utils import Conv1D

from typing import Optional, Tuple, Union

from modular_transformers.models.loss_utils import l2_reg, l1_reg, l2_curvature, l0_curvature, l1_curvature, curvature, l0_curvature_max, sparsity

loss_functions = {"l2_reg": l2_reg, "l1_reg": l1_reg, "l2_curvature": l2_curvature, "l0_curvature": l0_curvature, "l1_curvature": l1_curvature, "curvature": curvature, "l0_curvature_max": l0_curvature_max, "sparsity": sparsity}

class Transformer(transformers.GPT2Model):
    def __init__(self, config=None):
        super().__init__(config)
        self.h = nn.ModuleList()
        self.hook_functions = {}
        self.hook_losses = {}
        self.n_layer = config.n_layer
        self.hook_function_list = []
        self.attn_mask = None
        self.loss_hooks = config.loss_hooks
        dropout_dict = config.dropout_dict

        for layer in range(config.n_layer):
            #each block gets a new GPT2Config object
            #each block needs an output dimension to be given to 
            #it's dimension changer final layers
            #the last block needs to return the embedding dimension to the original size
            if layer == config.n_layer - 1:
                output_dim = config.n_embd
            else:
                output_dim = config.n_embds[layer+1]

            #variable dropout logic
            resid_pdrop = config.resid_pdrop
            attn_pdrop = config.attn_pdrop
            embd_pdrop = config.embd_pdrop

            if dropout_dict is not None and layer in dropout_dict:
                if "resid" in dropout_dict[layer]:
                    resid_pdrop = dropout_dict[layer]["resid"]
                if "attn" in dropout_dict[layer]:
                    attn_pdrop = dropout_dict[layer]["attn"]
                if "embd" in dropout_dict[layer]:
                    embd_pdrop = dropout_dict[layer]["embd"]
            
            block_config = transformers.GPT2Config(n_embd = config.n_embds[layer], 
                                                   hidden_size = config.n_embds[layer],
                                                   n_head = config.n_heads[layer], 
                                                   n_inner = config.n_inners[layer], 
                                                   activation_function = config.activation_functions[layer],
                                                   output_dim = output_dim,
                                                   resid_pdrop = resid_pdrop,
                                                   attn_pdrop = attn_pdrop,
                                                   embd_pdrop = embd_pdrop)
            block = TransformerBlock(block_config)
            self.h.append(block)

            # if layer in config.loss_hooks:
            #     block.ln_2.register_forward_hook(self.record_extra_loss_wrapper(layer))
            #     self.hook_functions[layer] = loss_functions[self.config.loss_hooks[layer]]
            #     self.hook_losses[layer] = 0
        
        if self.loss_hooks != None:
            self.set_hooks()
    
    def remove_hooks(self):
        for hook in self.hook_function_list:
            hook.remove()
        self.hook_function_list = []
            
    def set_hooks(self):
        for layer in range(self.n_layer):
            if layer in self.loss_hooks:
                block = self.h[layer]

                #currently using the output of ln_2 becuase it is residual before the MLP block
                # module = block.ln_2
                #residual after the MLP block is the output of the projection layer
                # module = block.mlp.c_proj
                module = block
                hook = module.register_forward_hook(self.record_extra_loss_wrapper(layer))
                self.hook_function_list.append(hook)

                self.hook_functions[layer] = loss_functions[self.loss_hooks[layer]]
                self.hook_losses[layer] = 0

    def record_extra_loss_wrapper(self, layer):
        def record_extra_loss(module, input, output):
            loss = self.hook_functions[layer](output, attn_mask = self.attn_mask)
            self.hook_losses[layer] += loss

        return record_extra_loss

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
        if config.output_dim != hidden_size:
            self.dim_changer = nn.Linear(hidden_size, config.output_dim, bias=False)
            for param in self.dim_changer.parameters():
                param.requires_grad = False
        else:
            self.dim_changer = None

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

        if self.dim_changer != None:
            hidden_states = self.dim_changer(hidden_states) #MY ADDITION ------------

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
    


class LM(transformers.GPT2LMHeadModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.transformer = Transformer(config)
        self.post_init()
    
    def forward_with_extra_loss(self, input, labels=None, attention_mask=None, attn_indices=None):
        self.transformer.attn_mask = attn_indices
        response = super().forward(input, labels=labels, attention_mask=attention_mask)
        return response

    def remove_hooks(self):
        self.transformer.remove_hooks()
    
    def set_hooks(self):
        self.transformer.set_hooks()

    def output_extra_losses(self):
        return self.transformer.output_extra_losses()
    
class ClassificationLM(transformers.GPT2ForSequenceClassification):
    def __init__(self, config=None):
        super().__init__(config)
        self.transformer = Transformer(config)
        self.pad_token_id = config.pad_token_id
        self.post_init()
    
    def forward_with_extra_loss(self, input, labels=None, attention_mask=None, attn_indices=None):
        self.transformer.attn_mask = attn_indices
        response = super().forward(input, labels=labels, attention_mask=attention_mask)
        return response

    def remove_hooks(self):
        self.transformer.remove_hooks()
    
    def set_hooks(self):
        self.transformer.set_hooks()

    def output_extra_losses(self):
        return self.transformer.output_extra_losses()