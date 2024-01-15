import torch

from pathlib import Path

from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed

from modular_transformers.models.gpt2.configuration_gpt2 import GPT2Config
from modular_transformers.models import components

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import orth
import pandas as pd
import json

from sklearn.decomposition import PCA

from tqdm import tqdm

import subprocess
import os

#set seeds
torch.manual_seed(0)
np.random.seed(0)
set_seed(0)

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#set tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

import gc

def load_data(num_bigrams):
    bigram_text = open("bigrams.txt", "r").read().split("\n")
    bigrams = [(int(b[b.find('\t')+1:]), b[:b.find('\t')]) for b in bigram_text if len(b) > 0]
    bigrams.sort(key=lambda x: x[0], reverse=True)
    bigrams = bigrams[:num_bigrams]

    bigram_tokens = [tokenizer.encode(b[1]) for b in bigrams]
    # bigram_tokens = [b + [tokenizer.pad_token_id]*(max_token_num-len(b)) for b in bigram_tokens]
    bigram_tokens = [torch.tensor(b) for b in bigram_tokens]
    # bigram_tokens = torch.cat(bigram_tokens, dim=0)
    return bigram_tokens

#big model
def load_model(model_path = "gpt2"):
    if model_path == "gpt2":
        orig_model = GPT2LMHeadModel.from_pretrained("gpt2")
    else:
        orig_model = components.LM.from_pretrained(model_path)
    return orig_model.to(device)

class Activations():
    def __init__(self, num_layers, max_token_num):
        self.activations = [[[] for _ in range(num_layers)] for _ in range(max_token_num)]
    
    def register_monitoring_hooks(self, model, max_token_num):
        def hook_wrapper(layer):
            def hook_function(module, input, output):
                input = input[0]
                for i in range(min(max_token_num, input.shape[1])):
                    self.activations[i][layer].append(input[:, i, :].detach().cpu())
            return hook_function

        for i, layer in enumerate(model.transformer.h):
            # layer.mlp.c_proj.register_forward_hook(hook_wrapper(i))
            # layer.attn.c_attn.register_forward_hook(hook_wrapper(i))
            layer.ln_2.register_forward_hook(hook_wrapper(i))

def make_activation_matrix(activations, max_token_num, num_layers):
    final = [[[] for _ in range(num_layers)] for _ in range(max_token_num)]
    for token_num, acts in enumerate(activations):
        if len(acts[0]) < 2:
            final = final[:token_num]
            max_token_num = token_num
            break
        for layer, layer_acts in enumerate(acts):
            matrix = torch.stack(layer_acts, dim=0).squeeze()
            final[token_num][layer] = matrix
    return final, max_token_num

def get_activation_matrix(model, bigram_tokens, max_token_num, num_layers):
    """
    :param model: model to get activations from
    :param bigram_tokens: list of bigram tokens
    :param max_token_num: max number of tokens in bigrams
    :param num_layers: number of layers in model
    :return: activation matrix: token_num: layer: activations (num_bigrams x hidden_size)
    """
    activation_class = Activations(num_layers, max_token_num)
    activation_class.register_monitoring_hooks(model, max_token_num)

    logits = []
    for bigram in tqdm(bigram_tokens):
        input = bigram.to(device)
        output = model(input)
        input.detach_()
        logits.append(output.logits[-1, :])
    logits = torch.stack(logits, dim=0)
    logits.detach_()
    activations = activation_class.activations

    activations, max_token_num = make_activation_matrix(activations, max_token_num, num_layers)

    return activations, max_token_num, logits

def get_pcas(activations, num_layers, max_token_num):
    pcas = [[None for _ in range(num_layers)] for _ in range(max_token_num)]
    for token_num, acts in enumerate(activations):
        for layer, matrix in enumerate(acts):
            pca = PCA()
            if not isinstance(matrix, list):
                matrix = matrix.detach().cpu().numpy()
            else:
                matrix = np.array(matrix).reshape(1, -1)
            pca.fit(matrix)
            pcas[token_num][layer] = pca
    return pcas

#find the number of components needed to explain a baseline variance as well as a universal cutoff
def get_num_components(orig_pcas, max_token_num, pca_dim, num_layers):
    num_components = [[{} for _ in range(num_layers)] for _ in range(max_token_num)]
    for token_num, layers_pcas in enumerate(orig_pcas):
        for layer, pca in enumerate(layers_pcas):
            variance = pca.explained_variance_ratio_
            for i, v in enumerate(variance.cumsum()):
                if v > 0.8:
                    num_components[token_num][layer]["explained_cutoff"] = i
                    break
            if len(variance) < pca_dim:
                num_components[token_num][layer]["cutoff"] = len(variance)
            else:
                num_components[token_num][layer]["cutoff"] = pca_dim
                
    return num_components

def get_orthogonal_vector(v):
    if np.all(v == 0):
        raise ValueError("The input vector cannot be the zero vector.")

    # Create a matrix with the input vector as the first row
    # and fill the rest with random values
    A = np.vstack([v, np.random.rand(len(v)-1, len(v))])

    # Use the null space to find a vector orthogonal to the input vector
    u = np.linalg.svd(A)[2][-1]

    return u

def register_pertubation_hooks(model, orig_pcas, num_components, pertubation_func, max_token_num, pertubation_layers):
    def hook_wrapper(pcs):
        def prehook(module, input):
            for i in range(min(max_token_num, input[0].shape[1], len(pcs))):
                pc_info = pcs[i]
                pc = pc_info[0]
                orthog_pc = pc_info[1]
                detached_input = input[0][:, i, :].detach().cpu().numpy()
                pertubation = pertubation_func(pc, detached_input, orthog_pc).to(device)
                input[0][:, i, :] += pertubation
                pertubation.detach_()
            return input
        return prehook

    #shuffle for different indexing
    reshuffled_pcas = {layer: {} for layer in pertubation_layers}
    for token_num, layers_pcas in enumerate(orig_pcas):
        for layer, pca in enumerate(layers_pcas):
            if layer in pertubation_layers:
                reshuffled_pcas[layer][token_num] = pca
    
    for layer, layer_pcas in reshuffled_pcas.items():
        pcs = []
        #for ever token num in the layer, find the average of the top num_components components
        for token_num, pca in layer_pcas.items():
            nc = num_components[token_num][layer]["cutoff"]
            pc = pca.components_[:nc, :]
            pc = pc.sum(axis=0) / nc
            pc = pc / np.linalg.norm(pc)
            orthog = get_orthogonal_vector(pc)
            pcs.append((pc, orthog))
        # model.transformer.h[layer].mlp.dropout.register_forward_pre_hook(hook_wrapper(pcs))
        # model.transformer.h[layer].attn.c_attn.register_forward_pre_hook(hook_wrapper(pcs))
        model.transformer.h[layer].ln_2.register_forward_pre_hook(hook_wrapper(pcs))

def complete_bigrams(model, bigrams):
    completions = []
    for bigram in bigrams:
        input = torch.tensor(tokenizer.encode(bigram)).to(device)
        output = model(input)
        prediction = output.logits[-1, :].argmax(dim=-1)
        tokenized = tokenizer.decode(prediction)
        completions.append(tokenized)
    return completions

def cosine_divergence(act1, act2):
    cosine_difs = torch.nn.functional.cosine_similarity(act1, act2)
    return cosine_difs.mean().item()

def distance_divergence(act1, act2):
    difference = act1 - act2
    dist = torch.norm(difference, p=2, dim=1).mean()
    return dist.item()

def get_KL_logit_divergence(perturbed_logits, orig_logits):
    KL_divergence = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(perturbed_logits, dim=-1), torch.nn.functional.softmax(orig_logits, dim=-1), reduction='batchmean')
    return KL_divergence.item()

def get_gpu_memory_usage():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], encoding='utf-8')
        # Convert the output into a list of integers for each GPU
        memory_usage = [int(x) for x in result.strip().split('\n')]
        return memory_usage
    except subprocess.CalledProcessError as e:
        print("Failed to run nvidia-smi: ", e)
        return []

def go(bigram_tokens, max_token_num, model_path, orig_activations, orig_pcas, orig_logits, pca_dim, pertubation_func, pertubation_size, perturbation_loc, name):
    layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 11]
    # name = "random"
    act_save_path = f"/om2/user/jackking/MyData/dynamics/activations/{perturbation_loc}/perturb_{pertubation_size}/{name}"
    save_path = f"/om2/user/jackking/modular_transformers/modular_transformers/dynamics/{perturbation_loc}/perturb_{pertubation_size}/{name}"
    if not os.path.exists(act_save_path):
        # Create the directory if it does not exist
        os.makedirs(act_save_path)
    if not os.path.exists(save_path):
        # Create the directory if it does not exist
        os.makedirs(save_path)

    bigram_results_df = pd.read_csv('/om2/user/jackking/modular_transformers/modular_transformers/dynamics/bigram_results.csv', index_col=0)
    test_bigrams = bigram_results_df['bigram'].tolist()

    bigram_results_df = pd.DataFrame(columns=['bigram'] + [layer for layer in layers])
    bigram_results_df['bigram'] = test_bigrams

    column_names = ["layer_num"] + ["cosine_lyapunov", "distance_lyapunov", "KL_logit_div"] + [f"cosine_sim_{i}" for i in layers] + [f"distance_{i}" for i in layers] 
    dataframes = [pd.DataFrame(columns=column_names) for _ in range(max_token_num)]

    # dataframes = [pd.read_csv(f'{save_path}/token_{token_num}_results_df.csv', index_col=0) for token_num in range(max_token_num)]

    for layer in layers:
        torch.cuda.empty_cache() 
        print(f'Running layer {layer}')
        gpu_memory_usage = get_gpu_memory_usage()
        print("GPU Memory Usage (in MB):", gpu_memory_usage)
        print(torch.cuda.memory_allocated())

        perturbed_model = load_model(model_path)
        num_layers = len(perturbed_model.transformer.h)
        num_components = get_num_components(orig_pcas, max_token_num, pca_dim, num_layers)

        register_pertubation_hooks(perturbed_model, orig_pcas, num_components, pertubation_func, max_token_num, [layer])
        perturbed_activations, max_token_num, perturbed_logits = get_activation_matrix(perturbed_model, bigram_tokens, max_token_num, num_layers)

        torch.save(perturbed_activations, f'{act_save_path}/{layer}_perturbed_activations.pt')

        completions = complete_bigrams(perturbed_model, test_bigrams)
        bigram_results_df[layer] = completions
        del perturbed_model
        gc.collect()
        torch.cuda.empty_cache() 

        KL_logit_div = get_KL_logit_divergence(perturbed_logits, orig_logits)

        for token_num in range(max_token_num):
            print(f'Running token {token_num}')
            if not layer == 11:
                cosine_difs = []
                distances = []
                for compare_layer in layers[layer+1:]:
                    cosine_dif = 1 - cosine_divergence(orig_activations[token_num][compare_layer], perturbed_activations[token_num][compare_layer])
                    distance = distance_divergence(orig_activations[token_num][compare_layer], perturbed_activations[token_num][compare_layer])
                    cosine_difs.append(cosine_dif)
                    distances.append(distance)
                cosine_lyapunov = np.log(np.abs(np.array(cosine_difs) + 1e-9 / pertubation_size)).sum() / len(cosine_difs)
                distance_lyapunov = np.log(np.array(distances) + 1e-9 / pertubation_size).sum() / len(distances)
                
                df = dataframes[token_num]
                new_df = pd.DataFrame({
                    "layer_num": [layer],
                    "cosine_lyapunov": [cosine_lyapunov],
                    "distance_lyapunov": [distance_lyapunov],
                    "KL_logit_div": [KL_logit_div],
                    **{f"cosine_sim_{i}": [cosine_difs[i - (layer + 1)]] for i in layers[layer+1:]},
                    **{f"distance_{i}": [distances[i - (layer + 1)]] for i in layers[layer+1:]}
                })
            else:
                df = dataframes[token_num]
                new_df = pd.DataFrame({
                    "layer_num": [layer],
                    "KL_logit_div": [KL_logit_div],
                })
            df = pd.concat([df, new_df], ignore_index=True)
            dataframes[token_num] = df
        
        for token_num, df in enumerate(dataframes):
            df.to_csv(f'{save_path}/token_{token_num}_results_df.csv')      
        bigram_results_df.to_csv(f'{save_path}/bigram_results.csv')  

def generate_random_pertubation(pertubation_size, pc):
    pertubation = torch.randn(pc.shape)
    pertubation = pertubation / np.linalg.norm(pertubation) * pertubation_size
    return pertubation
        

if __name__ == "__main__":
    num_bigrams = 10000
    bigram_tokens = load_data(num_bigrams)
    max_token_num = max([len(b) for b in bigram_tokens])
    model_path = 'gpt2'

    orig_model = load_model(model_path)
    num_layers = len(orig_model.transformer.h)
    orig_activations, max_token_num, orig_logits = get_activation_matrix(orig_model, bigram_tokens, max_token_num, num_layers)
    orig_pcas = get_pcas(orig_activations, num_layers, max_token_num)
    del orig_model
    gc.collect()
    torch.cuda.empty_cache() 

    pertubation_loc = "before_mlp"

    for pertubation_size in [0.01, 1, 10]:
        pca_dim = 1
        pertubation_func = lambda pc, input, orthog_pc: generate_random_pertubation(pertubation_size, pc)
        name = "random"
        go(bigram_tokens, max_token_num, model_path, orig_activations, orig_pcas, orig_logits, pca_dim, pertubation_func, pertubation_size, pertubation_loc, name)

        pca_dim = 20
        pertubation_func = lambda pc, input, orthog_pc: torch.tensor(orthog_pc) / np.linalg.norm(orthog_pc) * pertubation_size
        name = f"orthogonal_{pca_dim}pcs"
        go(bigram_tokens, max_token_num, model_path, orig_activations, orig_pcas, pca_dim, pertubation_func, pertubation_size, name)

        pca_dim = 5
        pertubation_func = lambda pc, input, orthog_pc: torch.tensor(orthog_pc) / np.linalg.norm(orthog_pc) * pertubation_size
        name = f"orthogonal_{pca_dim}pcs"
        go(bigram_tokens, max_token_num, model_path, orig_activations, orig_pcas, pca_dim, pertubation_func, pertubation_size, name)

        pca_dim = 20
        pertubation_func = lambda pc, input, orthog_pc: torch.tensor(pc) / np.linalg.norm(pc) * pertubation_size
        name = f"parallel{pca_dim}pcs"
        go(bigram_tokens, max_token_num, model_path, orig_activations, orig_pcas, pca_dim, pertubation_func, pertubation_size, name)

        pca_dim = 5
        pertubation_func = lambda pc, input, orthog_pc: torch.tensor(pc) / np.linalg.norm(pc) * pertubation_size
        name = f"parallel{pca_dim}pcs"
        go(bigram_tokens, max_token_num, model_path, orig_activations, orig_pcas, pca_dim, pertubation_func, pertubation_size, name)

        pca_dim = 20
        pertubation_func = lambda pc, input, orthog_pc: torch.tensor(pc) / np.linalg.norm(pc) * pertubation_size * -1
        name = f"negative{pca_dim}pcs"
        go(bigram_tokens, max_token_num, model_path, orig_activations, orig_pcas, pca_dim, pertubation_func, pertubation_size, name)

        pca_dim = 5
        pertubation_func = lambda pc, input, orthog_pc: torch.tensor(pc) / np.linalg.norm(pc) * pertubation_size * -1
        name = f"negative{pca_dim}pcs"
        go(bigram_tokens, max_token_num, model_path, orig_activations, orig_pcas, pca_dim, pertubation_func, pertubation_size, name)

