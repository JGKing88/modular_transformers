import logging
import sys
from collections import OrderedDict
import torch
import copy
from tqdm import tqdm
import numpy as np
from numpy.random.mtrand import RandomState

def permute_mat(mat):
    mat_flat = mat.flatten()
    assert(mat_flat.ndim==1)
    shuffle_idx = torch.randperm(mat_flat.shape[0])
    mat_flat_rnd = mat_flat[shuffle_idx]
    mat_perm = torch.reshape(mat_flat_rnd, mat.shape)
    return mat_perm

def initialize_gpt2_weights(model,mu=0,sigma=0.02,permute=False):
    model_perm = copy.deepcopy(model)
    orig_states = model_perm.state_dict()
    valid_keys=['attn.c_attn.weight','attn.c_attn.bias','attn.c_proj','ln','mlp','wte','wpe','lm_head']
    to_permute=np.sum([np.sum([valid_keys[n] in s for s in list(orig_states.keys())]) for n in range(len(valid_keys))])
    if permute:
        pbar=tqdm(total=to_permute,desc=f'permuting {to_permute} weights in {len(orig_states.keys())}')
    else:
        pbar = tqdm(total=to_permute, desc=f'initializing {to_permute} weights in {len(orig_states.keys())}')
    perm_states = dict.fromkeys(orig_states.keys())
    for key in orig_states.keys():
        if 'attn.c_attn.weight' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(a) if permute else permute_mat(b)
            pbar.update()
        elif 'attn.c_attn.bias' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(a) if permute else permute_mat(b)
            pbar.update()
        elif 'attn.c_proj' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(a) if permute else permute_mat(b)
            pbar.update()
        # modify layer norm
        elif 'ln' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(a) if permute else permute_mat(b)
            pbar.update()
        # modify feedforward layer
        elif 'mlp' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(a) if permute else permute_mat(b)
            pbar.update()
        elif 'wte' in key or 'wpe' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(a) if permute else permute_mat(b)
            pbar.update()
        elif 'lm_head' in key:
            a = orig_states[key]
            b = torch.normal(mu, sigma, size=a.shape)
            perm_states[key] = permute_mat(a) if permute else permute_mat(b)
            pbar.update()
        else:
            perm_states[key] = orig_states[key]
    return perm_states

