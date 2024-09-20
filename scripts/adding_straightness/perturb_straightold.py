import torch
import math

from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed, AutoModel, AutoConfig
from datasets import load_dataset

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from transformers import BatchEncoding

from minicons import scorer
from torch.utils.data import DataLoader

import json
from datasets import load_dataset, load_from_disk
import os
from tqdm import tqdm

import pickle
import gc
from modular_transformers.straightening.straightening_utils import compute_model_activations, compute_model_curvature
from modular_transformers.models.gpt2.configuration_gpt2 import GPT2Config

from modular_transformers.models import components
from transformer_xray.perturb_utils import register_pertubation_hooks

from torchviz import make_dot

from functools import partial

max_len = 25
layer_num = 48
embedding_size = 1600
first_sequence_len = 4

num_perturbations = 1

#set seed
set_seed(42)

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#set tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix
torch.set_grad_enabled(False)


def get_orthogonal_vectors(directions):
    print(directions.shape)
    for i in range(0, directions.shape[1], 2):
        directions[:, i], directions[:, i+1] = directions[:, i+1], -directions[:, i]
    return directions

def perturb_input_random(input, hook, multiplier):
    perturb_idx = input.shape[1] - 1
    norm = (input[:, perturb_idx, :] - input[:, perturb_idx-1, :]).norm(dim=-1, keepdim=True)
    perturbation = torch.randn_like(input[:, perturb_idx, :])
    unit_perturbation = perturbation / perturbation.norm(dim=-1, keepdim=True)
    input[:, perturb_idx, :] = input[:, perturb_idx, :] + unit_perturbation * norm * multiplier
    print(f"random size: {norm.mean() * multiplier}")
    return input

def perturb_input_parallel(input, hook,  multiplier):
    perturb_idx = input.shape[1] - 1
    directions = input[:, perturb_idx, :] - input[:, perturb_idx-1, :]
    input[:, perturb_idx, :] = input[:, perturb_idx, :] + directions * multiplier
    print(f"parallel size: {directions.norm(dim=-1).mean() * multiplier}")
    return input

def perturb_input_prev_parallel(input, hook, multiplier):
    perturb_idx = input.shape[1] - 1
    directions = input[:, perturb_idx-1, :] - input[:, perturb_idx-2, :]
    input[:, perturb_idx, :] = input[:, perturb_idx, :] + directions * multiplier
    print(f"prev parallel size: {directions.norm(dim=-1).mean() * multiplier}")
    return input

def perturb_input_straighten(input, hook, multiplier):
    perturb_idx = input.shape[1] - 1
    P2 = input[:, perturb_idx-1, :].clone().detach()
    P1 = input[:, perturb_idx-2, :].clone().detach()
    Q = input[:, perturb_idx, :].clone().detach()

    D = P2 - P1
    Q_P1 = Q - P1
    proj_d_q_p1 = torch.sum(D * Q_P1, dim=-1, keepdim=True) / torch.sum(D * D, dim=-1, keepdim=True) * D
    perturbation = Q_P1 - proj_d_q_p1
    input[:, perturb_idx, :] = input[:, perturb_idx, :] + perturbation * multiplier
    print(f"straighten size: {perturbation.norm(dim=-1).mean() * multiplier}")
    return input

def perturb_input_orthog(input, hook, multiplier):
    perturb_idx = input.shape[1] - 1
    directions = input[:, perturb_idx, :] - input[:, perturb_idx-1, :]
    orthogonal_directions = get_orthogonal_vectors(directions)
    input[:, perturb_idx, :] = input[:, perturb_idx, :] + orthogonal_directions * multiplier
    print(f"orthogonal size: {orthogonal_directions.norm(dim=-1).mean() * multiplier}")
    return input

def perturb_input_on_path(input, hook, multiplier):
    perturb_idx = input.shape[1] - 1
    P2 = input[:, perturb_idx-1, :].clone().detach()
    P1 = input[:, perturb_idx-2, :].clone().detach()
    Q = input[:, perturb_idx, :].clone().detach()

    goal_point = (P2 - P1) + P2
    perturbation = goal_point - Q
    input[:, perturb_idx, :] = input[:, perturb_idx, :] + perturbation * multiplier
    print(f"on path size: {perturbation.norm(dim=-1).mean() * multiplier}")
    return input

def generate_perturbed_token(model, data, perturb_function, perturb_location):
    model.reset_hooks()

    sequence_len = data.shape[1]

    post_activations = torch.zeros((len(data), layer_num, sequence_len, embedding_size))
    def record_post_activations(input, hook, layer):
        post_activations[:, layer, :, :] = input

    # mid_activations = torch.zeros((len(data), layer_num, sequence_len, embedding_size))
    # def record_mid_activations(input, hook, layer):
    #     mid_activations[:, layer, :, :] = input

    fwd_hooks = []
    for layer in range(layer_num):
        fwd_hooks.append((utils.get_act_name("resid_post", layer), partial(record_post_activations, layer=layer)))
        # fwd_hooks.append((utils.get_act_name("resid_mid", layer), partial(record_mid_activations, layer=layer)))

    fwd_hooks.append((
            perturb_location,
            perturb_function
        ))

    logits = model.run_with_hooks(
        data, 
        return_type="logits", 
        fwd_hooks=fwd_hooks,
    )
    model.reset_hooks()

    # activations = {"post": post_activations, "mid": mid_activations}
    activations = {"post": post_activations}

    new_token = logits.argmax(dim=-1)[:, -1]

    return new_token, activations


def generate_sentences(data, perturb_function, perturb_location):

    model = HookedTransformer.from_pretrained("gpt2-xl", device=device)

    perturbed_data = data.clone()
    activations = [None] * num_perturbations

    for i in range(num_perturbations):
        new_token, activations[i] = generate_perturbed_token(model, perturbed_data, perturb_function, perturb_location)
        perturbed_data = torch.cat([perturbed_data, new_token.unsqueeze(1)], dim=1)
        torch.cuda.empty_cache()

    #generate new sentences by adding new token to the end of the sentence
    for i in range(max_len - first_sequence_len - num_perturbations):
        logits = model(perturbed_data)
        new_token = logits.argmax(dim=-1)[:, -1]
        perturbed_data = torch.cat([perturbed_data, new_token.unsqueeze(1)], dim=1)
        torch.cuda.empty_cache()

    return perturbed_data, activations

def generate_curved_sentences(data, multiplier, perturb_location):
    perturb_function = partial(perturb_input_straighten, multiplier=multiplier)
    return generate_sentences(data, perturb_function, perturb_location)

def generate_straightened_sentences(data, multiplier, perturb_location):
    perturb_function = partial(perturb_input_straighten, multiplier=multiplier)
    return generate_sentences(data, perturb_function, perturb_location)

def generate_parallel_sentences(data, multiplier, perturb_location):
    perturb_function = partial(perturb_input_parallel, multiplier=multiplier)
    return generate_sentences(data, perturb_function, perturb_location)
    
def generate_orthogonal_sentences(data, multiplier, perturb_location):
    perturb_function = partial(perturb_input_orthog, multiplier=multiplier)
    return generate_sentences(data, perturb_function, perturb_location)

def generate_random_sentences(data, multiplier, perturb_location):
    perturb_function = partial(perturb_input_random, multiplier=multiplier)
    return generate_sentences(data, perturb_function, perturb_location)

def generate_prev_parallel_sentences(data, multiplier, perturb_location):
    perturb_function = partial(perturb_input_prev_parallel, multiplier=multiplier)
    return generate_sentences(data, perturb_function, perturb_location)

def generate_on_path_sentences(data, multiplier, perturb_location):
    perturb_function = partial(perturb_input_on_path, multiplier=multiplier)
    return generate_sentences(data, perturb_function, perturb_location)

def record_activations(model, data):
    post_activations = torch.zeros((len(data), layer_num, max_len, embedding_size))
    def record_post_activations(input, hook, layer):
        post_activations[:, layer, :, :] = input

    # mid_activations = torch.zeros((len(data), layer_num, max_len, embedding_size))
    # def record_mid_activations(input, hook, layer):
    #     mid_activations[:, layer, :, :] = input

    fwd_hooks = []
    for layer in range(layer_num):
        fwd_hooks.append((utils.get_act_name("resid_post", layer), partial(record_post_activations, layer=layer)))
        # fwd_hooks.append((utils.get_act_name("resid_mid", layer), partial(record_mid_activations, layer=layer)))

    model.run_with_hooks(
        data, 
        return_type=None, 
        fwd_hooks=fwd_hooks,
    )
    model.reset_hooks()

    # return {"post": post_activations, "mid": mid_activations}
    return {"post": post_activations}


def run_perturbed(gen_data, gen_activations):

    #gen activations shape: (num_sentences, num_layers, num_tokens, hidden_size)
    gen_curvatures = [{}] * num_perturbations

    for i in range(num_perturbations):
        gen_curvatures[i]["post"] = compute_model_curvature(gen_activations[i]["post"])
        # gen_curvatures[i]["mid"] = compute_model_curvature(orthogonal_gen_activations[i]["mid"])
        
    #get curvature with sentences
    model = HookedTransformer.from_pretrained("gpt2-xl", device=device)
    data_activations = record_activations(model,gen_data)
    data_curvature = {}
    data_curvature["post"] = compute_model_curvature(data_activations["post"])
    # data_curvature["mid"] = compute_model_curvature(data_activations["mid"])

    #get surprisal with sentences
    # batch_size = 100
    # data_batched_tensor = gen_data.view(-1, batch_size, gen_data.shape[1])
    # attn_mask = torch.ones(data_batched_tensor.shape[0], data_batched_tensor.shape[1], data_batched_tensor.shape[2])
    # data_batched_tensor = BatchEncoding({"input_ids": data_batched_tensor, "attention_mask": attn_mask})
    data_decoded = [tokenizer.decode(sentence) for sentence in gen_data]

    model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(device)
    model = scorer.IncrementalLMScorer(model, tokenizer=tokenizer, device=device)
    batch_size = 1000
    for i in range(0, len(data_decoded), batch_size):
        data_decoded_batch = data_decoded[i:i+batch_size]
        surprisals = torch.tensor(model.sequence_score(data_decoded_batch, reduction = lambda x: -x.sum(0)))
        if i == 0:
            surprisals_all = surprisals
        else:
            surprisals_all = torch.cat([surprisals_all, surprisals], dim=0)

    return_dict = {
        "surprisals": surprisals_all,
        "sentences": data_decoded,
        "curvatures": data_curvature,
        "gen_curvatures": gen_curvatures
    }
    return return_dict

def launch(data, perturb_location):
    
    perturb_funcs_dict = {
        "parallel": generate_parallel_sentences,
        "orthogonal": generate_orthogonal_sentences,
        "random": generate_random_sentences,
        "straightened": generate_straightened_sentences,
        "curved": generate_curved_sentences,
        "prev_parallel": generate_prev_parallel_sentences,
        "neg_prev_parallel": generate_prev_parallel_sentences,
        "on_path": generate_on_path_sentences,
        "off_path": generate_on_path_sentences
    }

    path_to_dict = f"/om2/user/jackking/modular_transformers/scripts/adding_straightness/perturbedx{num_perturbations}_nocontext_perturb_straight_results_{perturb_location}.pkl"
    # if os.path.exists(path_to_dict):
    #     new_surprisals = pickle.load(open(path_to_dict, "rb"))
    # else:
    new_surprisals = {}
        
    data = data[:5000]
    cut_data = data[:, :first_sequence_len].to(device)

    reg_func = lambda input, hook: None
    gen_data, gen_activations = generate_sentences(cut_data, reg_func, perturb_location)

    new_surprisals = {"normal": run_perturbed(gen_data, gen_activations)}

    for multiplier in [0.1, 0.25, 0.5, 1, 2, 3]:
        new_surprisals[multiplier] = {}
        for perturb_type in ["parallel", "orthogonal", "random", "straightened", "curved", "prev_parallel", "neg_prev_parallel", "on_path", "off_path"]:

            perturb_type_gen_func = perturb_funcs_dict[perturb_type]

            if perturb_type == "neg_prev_parallel" or perturb_type == "off_path" or perturb_type == "curved":
                gen_data, gen_activations = perturb_type_gen_func(cut_data, -multiplier, perturb_location)
            else:
                gen_data, gen_activations = perturb_type_gen_func(cut_data, multiplier, perturb_location)
                
            new_surprisals[multiplier][perturb_type] = run_perturbed(gen_data, gen_activations)

    with open(path_to_dict, 'wb') as f:
        pickle.dump(new_surprisals, f)
    


if __name__ == "__main__":
    data_dir = "/rdma/vast-rdma/vast/evlab/ehoseini/MyData/sent_sampling/analysis/straightening/generation/sentences_ud_sentencez_token_filter_v3_textNoPeriod_cntx_3_cont_7.pkl"
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
    tokenizer.pad_token = tokenizer.eos_token
    data = tokenizer.batch_encode_plus(data, add_special_tokens=True, padding='longest', return_tensors="pt")["input_ids"]

    perturb_location = "blocks.15.hook_resid_post"
    launch(data, perturb_location)

    perturb_location = "blocks.15.hook_resid_mid"
    launch(data, perturb_location)

    perturb_location = "blocks.5.hook_resid_mid"
    launch(data, perturb_location)
    perturb_location = "blocks.5.hook_resid_post"
    launch(data, perturb_location)
    
    perturb_location = "blocks.30.hook_resid_mid"
    launch(data, perturb_location)
    perturb_location = "blocks.30.hook_resid_post"
    launch(data, perturb_location)
    