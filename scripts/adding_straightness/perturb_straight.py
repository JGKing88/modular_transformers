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
from modular_transformers.straightening.straightening_utils import (
    compute_model_activations,
    compute_model_curvature,
)
from modular_transformers.models.gpt2.configuration_gpt2 import GPT2Config

from modular_transformers.models import components
from transformer_xray.perturb_utils import register_pertubation_hooks

from torchviz import make_dot

from functools import partial

# Configuration constants
max_len = 25
layer_num = 48
embedding_size = 1600
first_sequence_len = 4

# Set dimensions for perturbation analysis
principal_dimensions_for_curved = 10
principal_dimensions_for_straight = 1600

# set seed
set_seed(42)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix

torch.set_grad_enabled(False)


def perturb_input(input, hook, perturbations, principal_dimensions):
    perturb_idx = input.shape[1] - 1

    # input_flattened = input.view(-1, 1600)
    # input_centered = input_flattened - input_flattened.mean(dim=0, keepdim=True)
    # U, S, V = torch.linalg.svd(input_centered, full_matrices=False)
    # principal_components = V.T[:, :principal_dimensions]
    # input_centered = input_centered.view(input.shape[0], input.shape[1], input.shape[2])
    # projected_input = torch.matmul(input_centered, principal_components)
    # new_points = projected_input[:, perturb_idx, :] + perturbations.to(device)
    # new_points = torch.matmul(new_points, principal_components.T)
    # input[:, perturb_idx, :] = new_points

    input[:, perturb_idx, :] = input[:, perturb_idx, :] + perturbations.to(device)

    return input


def get_curvature(P1, P2, P3):
    v1 = P2 - P1
    v2 = P3 - P2
    v1 = v1 / v1.norm(dim=-1, keepdim=True)
    v2 = v2 / v2.norm(dim=-1, keepdim=True)
    curvature = torch.acos(torch.sum(v1 * v2, dim=-1))
    return curvature


def get_perturbations(model, data, perturb_location, size):
    model.reset_hooks()

    num_samples = 100
    principal_dimensions = 10

    straightest_perturbations = torch.zeros(len(data), embedding_size)
    curviest_perturbations = torch.zeros(len(data), embedding_size)

    def record_perturbation_directions(input, hook):
        cloned_input = input.clone()
        input_flattened = input.view(-1, 1600)
        input_centered = input_flattened - input_flattened.mean(dim=0, keepdim=True)
        U, S, V = torch.linalg.svd(input_centered, full_matrices=False)
        principal_components = V.T[:, :principal_dimensions]

        input_centered = input_centered.view(
            input.shape[0], input.shape[1], input.shape[2]
        )
        projected_input = torch.matmul(input_centered, principal_components)

        perturb_idx = input.shape[1] - 1
        random_directions = (
            torch.FloatTensor(num_samples, input.shape[0], principal_dimensions)
            .uniform_(-1, 1)
            .to(device)
            * size
            * 10
        )
        norm = projected_input[:, perturb_idx, :].norm(dim=-1, keepdim=True)
        # random_perturbations = (random_directions / random_directions.norm(dim=-1, keepdim=True)) * norm * size * 10
        random_perturbations = torch.matmul(random_directions, principal_components.T)

        # random_perturbations = random_directions
        # new_points = projected_input[:, perturb_idx, :] + random_perturbations
        # new_points = torch.matmul(new_points, principal_components.T)
        new_points = input[:, perturb_idx, :] + random_perturbations
        # norm = input[:, perturb_idx, :].norm(dim=-1, keepdim=True)
        # new_points = new_points / new_points.norm(dim=-1, keepdim=True) * norm.view(1, -1, 1)

        perturbations_curves = torch.zeros(num_samples, input.shape[0]).to(device)

        perturbations_curves = torch.zeros(num_samples, input.shape[0]).to(device)
        for i, new_point in enumerate(new_points):
            curvature = get_curvature(
                cloned_input[:, perturb_idx - 2, :],
                cloned_input[:, perturb_idx - 1, :],
                new_point,
            )
            perturbations_curves[i] = curvature

        min_indices = torch.argmin(perturbations_curves, dim=0)
        for i in range(input.shape[0]):
            straightest_perturbations[i, :] = random_perturbations[min_indices[i], i, :]

        max_indices = torch.argmax(perturbations_curves, dim=0)
        for i in range(input.shape[0]):
            curviest_perturbations[i, :] = random_perturbations[max_indices[i], i, :]

    fwd_hooks = [(perturb_location, record_perturbation_directions)]

    model.run_with_hooks(
        data,
        return_type=None,
        fwd_hooks=fwd_hooks,
    )

    model.reset_hooks()

    if size >= 1:
        straightest_perturbations = straightest_perturbations / 2

    return straightest_perturbations, curviest_perturbations


def generate_perturbed_token(model, data, perturb_function):

    sequence_len = data.shape[1]

    post_activations = torch.zeros((len(data), layer_num, sequence_len, embedding_size))

    def record_post_activations(input, hook, layer):
        post_activations[:, layer, :, :] = input

    # mid_activations = torch.zeros((len(data), layer_num, sequence_len, embedding_size))
    # def record_mid_activations(input, hook, layer):
    #     mid_activations[:, layer, :, :] = input

    fwd_hooks = []

    fwd_hooks.append((perturb_location, perturb_function))

    for layer in range(layer_num):
        fwd_hooks.append(
            (
                utils.get_act_name("resid_post", layer),
                partial(record_post_activations, layer=layer),
            )
        )
        # fwd_hooks.append((utils.get_act_name("resid_mid", layer), partial(record_mid_activations, layer=layer)))

    logits = model.run_with_hooks(data, return_type="logits", fwd_hooks=fwd_hooks)

    # activations = {"post": post_activations, "mid": mid_activations}
    activations = {"post": post_activations}

    # new_token = logits.argmax(dim=-1)[:, -1]

    first_tokens = data[:, 0].unsqueeze(1)
    new_tokens = logits.argmax(dim=-1)
    new_tokens = torch.cat([first_tokens, new_tokens], dim=1)

    return new_tokens, activations


def continued_gen(model, data, length):
    # generate new sentences by adding new token to the end of the sentence
    final_data = torch.zeros((len(data), max_len), dtype=torch.int64).to(device)

    batch_size = 500
    batch_indxs = torch.arange(0, len(data), batch_size)
    for i in range(len(batch_indxs) - 1):
        batch = data[batch_indxs[i] : batch_indxs[i + 1]]

        for _ in range(length):
            first_tokens = batch[:, 0].unsqueeze(1)

            logits = model(batch).logits

            # new_token = logits.argmax(dim=-1)[:, -1]
            # batch = torch.cat([batch, new_token.unsqueeze(1)], dim=1)

            batch = logits.argmax(dim=-1)
            batch = torch.cat([first_tokens, batch], dim=1)

            torch.cuda.empty_cache()

        final_data[batch_indxs[i] : batch_indxs[i + 1], :] = batch.type(torch.int64)

    return final_data


def generate_normal_sentences(data):
    model = HookedTransformer.from_pretrained("gpt2-xl", device=device)

    normal_activations = []
    normal_data = data.clone()

    for i in range(num_perturbations):
        # new_token, activations = generate_perturbed_token(model, normal_data, perturb_function = lambda input, hook: None)
        # normal_data = torch.cat([normal_data, new_token.unsqueeze(1)], dim=1)
        normal_data, activations = generate_perturbed_token(
            model, normal_data, perturb_function=lambda input, hook: None
        )
        normal_activations.append(activations)
        gc.collect()
        torch.cuda.empty_cache()

    del model

    model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(device)
    model.output_logits = True

    length = max_len - first_sequence_len - num_perturbations
    normal_data = continued_gen(model, normal_data, length)

    return normal_data, normal_activations


def generate_sentences(data, perturb_location, size):
    """
    Generate perturbed sentences using GPT2-XL:
    - Creates both straighter and curved versions
    - Applies perturbations at specified locations
    - Handles memory management with garbage collection
    """
    model = HookedTransformer.from_pretrained("gpt2-xl", device=device)

    straighter_activations = []
    curved_activations = []

    straighter_data = data.clone()
    curved_data = data.clone()

    for i in range(num_perturbations):
        straightest_perturbations, curviest_perturbations = get_perturbations(
            model, straighter_data, perturb_location, size
        )
        perturb_function = partial(
            perturb_input,
            perturbations=straightest_perturbations,
            principal_dimensions=principal_dimensions_for_straight,
        )
        # new_token, activations = generate_perturbed_token(model, straighter_data, perturb_function)
        # straighter_data = torch.cat([straighter_data, new_token.unsqueeze(1)], dim=1)
        straighter_data, activations = generate_perturbed_token(
            model, straighter_data, perturb_function
        )
        straighter_activations.append(activations)

        perturb_function = partial(
            perturb_input,
            perturbations=curviest_perturbations,
            principal_dimensions=principal_dimensions_for_curved,
        )
        # new_token, activations = generate_perturbed_token(model, curved_data, perturb_function)
        # curved_data = torch.cat([curved_data, new_token.unsqueeze(1)], dim=1)
        curved_data, activations = generate_perturbed_token(
            model, curved_data, perturb_function
        )
        curved_activations.append(activations)

        gc.collect()
        torch.cuda.empty_cache()

    del model

    model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(device)
    model.output_logits = True

    length = max_len - first_sequence_len - num_perturbations
    straighter_data = continued_gen(model, straighter_data, length)
    gc.collect()
    torch.cuda.empty_cache()
    curved_data = continued_gen(model, curved_data, length)
    gc.collect()
    torch.cuda.empty_cache()

    return straighter_data, curved_data, straighter_activations, curved_activations


def record_activations(model, data):
    post_activations = torch.zeros((len(data), layer_num, max_len, embedding_size))

    def record_post_activations(input, hook, layer):
        post_activations[:, layer, :, :] = input

    # mid_activations = torch.zeros((len(data), layer_num, max_len, embedding_size))
    # def record_mid_activations(input, hook, layer):
    #     mid_activations[:, layer, :, :] = input

    fwd_hooks = []
    for layer in range(layer_num):
        fwd_hooks.append(
            (
                utils.get_act_name("resid_post", layer),
                partial(record_post_activations, layer=layer),
            )
        )
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
    # gen activations shape: (num_sentences, num_layers, num_tokens, hidden_size)
    gen_curvatures = [{}] * num_perturbations

    for i in range(num_perturbations):
        gen_curvatures[i]["post"] = compute_model_curvature(gen_activations[i]["post"])
        # gen_curvatures[i]["mid"] = compute_model_curvature(gen_activations[i]["mid"])

    # get curvature with sentences
    model = HookedTransformer.from_pretrained("gpt2-xl", device=device)
    data_activations = record_activations(model, gen_data)
    data_curvature = {}
    data_curvature["post"] = compute_model_curvature(data_activations["post"])
    # data_curvature["mid"] = compute_model_curvature(data_activations["mid"])

    # get surprisal with sentences
    data_decoded = [tokenizer.decode(sentence) for sentence in gen_data]

    model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(device)
    model = scorer.IncrementalLMScorer(model, tokenizer=tokenizer, device=device)
    batch_size = 1000
    for i in range(0, len(data_decoded), batch_size):
        data_decoded_batch = data_decoded[i : i + batch_size]
        surprisals = torch.tensor(
            model.sequence_score(data_decoded_batch, reduction=lambda x: -x.sum(0))
        )
        if i == 0:
            surprisals_all = surprisals
        else:
            surprisals_all = torch.cat([surprisals_all, surprisals], dim=0)

    return_dict = {
        "surprisals": surprisals_all,
        "sentences": data_decoded,
        "curvatures": data_curvature,
        "gen_curvatures": gen_curvatures,
    }
    return return_dict


def launch(data, perturb_location):
    """
    Main experiment launcher:
    - Handles different perturbation sizes
    - Generates and analyzes sentences
    - Saves results to disk
    """
    path_to_dict = f"/om2/user/jackking/modular_transformers/scripts/adding_straightness/data/perturb_{num_perturbations}x_straight_results_{perturb_location}_full_gen.pkl"
    if os.path.exists(path_to_dict):
        new_surprisals = pickle.load(open(path_to_dict, "rb"))
    else:
        new_surprisals = {}

    data = data[:5000]
    cut_data = data[:, :first_sequence_len].to(device)

    normal_data, normal_activations = generate_normal_sentences(cut_data)
    normal_results = run_perturbed(normal_data, normal_activations)
    print("normal analyzed")

    new_surprisals["normal"] = normal_results

    for size in [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]:
        if size in new_surprisals:
            continue
        print(size)
        straighter_data, curved_data, straighter_activations, curved_activations = (
            generate_sentences(cut_data, perturb_location, size)
        )
        print("sentences generated")

        straighter_results = run_perturbed(straighter_data, straighter_activations)
        print("straighter analyzed")
        curved_results = run_perturbed(curved_data, curved_activations)
        print("curved analyzed")

        new_surprisals[size] = {
            "straighter": straighter_results,
            "curved": curved_results,
        }

        with open(path_to_dict, "wb") as f:
            pickle.dump(new_surprisals, f)


if __name__ == "__main__":
    data_dir = "/rdma/vast-rdma/vast/evlab/ehoseini/MyData/sent_sampling/analysis/straightening/generation/sentences_ud_sentencez_token_filter_v3_textNoPeriod_cntx_3_cont_7.pkl"
    with open(data_dir, "rb") as f:
        data = pickle.load(f)
    tokenizer.pad_token = tokenizer.eos_token
    data = tokenizer.batch_encode_plus(
        data, add_special_tokens=True, padding="longest", return_tensors="pt"
    )["input_ids"]

    num_perturbations = 1

    perturb_location = "blocks.15.hook_resid_post"
    launch(data, perturb_location)
    print("15 post done")

    perturb_location = "blocks.5.hook_resid_post"
    launch(data, perturb_location)
    print("5 post done")

    perturb_location = "blocks.20.hook_resid_post"
    launch(data, perturb_location)
    print("5 post done")

    perturb_location = "blocks.30.hook_resid_post"
    launch(data, perturb_location)
    print("30 post done")

    num_perturbations = 3

    perturb_location = "blocks.15.hook_resid_post"
    launch(data, perturb_location)
    print("15 post done")

    perturb_location = "blocks.5.hook_resid_post"
    launch(data, perturb_location)
    print("5 post done")

    perturb_location = "blocks.20.hook_resid_post"
    launch(data, perturb_location)
    print("5 post done")

    perturb_location = "blocks.30.hook_resid_post"
    launch(data, perturb_location)
    print("30 post done")
