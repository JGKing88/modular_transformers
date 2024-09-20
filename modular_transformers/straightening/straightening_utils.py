

import os
import numpy as np
import sys
from pathlib import Path
import torch
from tqdm import tqdm

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


# testing the curvature
# just testing whay its not working
def compute_model_activations(model,indexed_tokens,device):
    # get activations
    model.eval()
    all_layers = []
    for i in tqdm(range(len(indexed_tokens))):
        try:
            tokens_tensor = torch.tensor([indexed_tokens[i]]).to(device)
        except:
            tokens_tensor = torch.tensor(indexed_tokens[i]).to(device)
        with torch.no_grad():
            outputs = model(tokens_tensor, output_hidden_states=True, output_attentions=False)
            hidden_states = outputs['hidden_states']
            # squeeze the first dimension
            hidden_states = [x.squeeze(0).cpu() for x in hidden_states]
        all_layers.append(hidden_states)
    torch.cuda.empty_cache()
    return all_layers

def compute_model_curvature(all_layers):
    #all_layers shape: (samples, layers, tokens, hidden_size)
    all_layer_curve = []
    all_layer_curve_all = []
    all_layer_curve_rnd = []
    all_layer_curve_rnd_all = []
    for idk, layer_act in tqdm(enumerate(all_layers)):
        True
        sent_act = [torch.diff(x, axis=0).cpu() for x in layer_act]
        sent_act = [normalized(x) for x in sent_act]
        curvature = []
        for idy, vec in (enumerate(sent_act)):
            curve = [np.dot(vec[idx, :], vec[idx + 1, :]) for idx in range(vec.shape[0] - 1)]
            curvature.append(np.arccos(curve))
        all_layer_curve.append([np.mean(x) for x in curvature])
        all_layer_curve_all.append(curvature)

    curve_ = np.stack(all_layer_curve).transpose()
    curve_change = (curve_[1:, :] - curve_[1, :])
    # make a dictionary with fieldds 'curve','curve_change','all_layer_curve_all' and return the dictionary
    torch.cuda.empty_cache()
    return dict(curve=curve_,curve_change=curve_change,all_layer_curve_all=all_layer_curve_all)





