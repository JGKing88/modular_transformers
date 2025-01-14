import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util

from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from minicons import scorer

import json
from datasets import load_dataset, load_from_disk
import os
from tqdm import tqdm

from scipy import stats


import pickle
from modular_transformers.straightening.straightening_utils import (
    compute_model_activations,
    compute_model_curvature,
)
from modular_transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import AutoTokenizer, AutoModelForCausalLM

from torchviz import make_dot

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

path = "/om2/user/jackking/modular_transformers/scripts/adding_straightness/data"


def get_sentences():
    save_path = f"{path}/gpt-4_outputs.pkl"
    gpt4_sentences = pickle.load(open(save_path, "rb"))

    first_sequence_len = 4
    max_len = 25
    client = OpenAI(api_key="")

    data_dir = "/rdma/vast-rdma/vast/evlab/ehoseini/MyData/sent_sampling/analysis/straightening/generation/sentences_ud_sentencez_token_filter_v3_textNoPeriod_cntx_3_cont_7.pkl"
    with open(data_dir, "rb") as f:
        data = pickle.load(f)

    for i, sentence in enumerate(data[:5000]):
        sentence_start = tokenizer.encode(sentence)[0:first_sequence_len]
        sentence_start = tokenizer.decode(sentence_start)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "continue the following sentence: "},
                {"role": "user", "content": sentence_start},
            ],
            max_tokens=21,
        )

        completion = sentence_start + response.choices[0].message.content
        gpt4_sentences.append(completion)

        if i % 50 == 0:
            print(f"Completed {i} sentences.")

            with open(save_path, "wb") as f:
                pickle.dump(gpt4_sentences, f)

    with open(save_path, "wb") as f:
        pickle.dump(gpt4_sentences, f)


def compare(num_perturbations, perturb_location):
    save_path = f"{path}/gpt-4_outputs.pkl"

    with open(save_path, "rb") as f:
        gpt4_sentences = pickle.load(f)

    path_to_dict = f"/om2/user/jackking/modular_transformers/scripts/adding_straightness/data/perturb_{num_perturbations}x_straight_results_{perturb_location}.pkl"
    # path_to_dict = f"/om2/user/jackking/modular_transformers/scripts/adding_straightness/perturb_straight_byact_results_{perturb_location}.pkl"

    results = pickle.load(open(path_to_dict, "rb"))

    save_path = (
        f"{path}/gpt-4_similarities_perturb_{num_perturbations}x_{perturb_location}.pkl"
    )
    # save_path = f"{path}/gpt-4_similarities_perturb_{perturb_location}_byact.pkl"

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    gpt_embeddings = model.encode(gpt4_sentences, convert_to_tensor=True)

    new_dict = {}

    normal_sentences = results["normal"]["sentences"]

    perturbed_embeddings = model.encode(normal_sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(gpt_embeddings, perturbed_embeddings)
    new_dict["normal"] = cosine_scores

    with open(save_path, "wb") as f:
        pickle.dump(new_dict, f)

    types = [
        "parallel",
        "orthogonal",
        "random",
        "straightened",
        "curved",
        "prev_parallel",
        "neg_prev_parallel",
        "on_path",
        "off_path",
    ]
    multipliers = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]

    types = ["straighter", "curved"]
    multipliers = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]

    for multiplier in multipliers:
        print(multiplier)
        new_dict[multiplier] = {}
        for perturb_type in types:
            sentences = results[multiplier][perturb_type]["sentences"]
            # find indices of sentences that are different from normal
            indices = [i for i, x in enumerate(sentences) if x != normal_sentences[i]]
            sentences = [sentences[i] for i in indices]
            selected_gpt_embeddings = torch.index_select(
                gpt_embeddings, 0, torch.tensor(indices).to(device)
            )
            perturbed_embeddings = model.encode(sentences, convert_to_tensor=True)
            cosine_scores = util.cos_sim(selected_gpt_embeddings, perturbed_embeddings)
            new_dict[multiplier][perturb_type] = cosine_scores

        with open(save_path, "wb") as f:
            pickle.dump(new_dict, f)


if __name__ == "__main__":
    num_perturbations = 1
    perturb_location = "blocks.5.hook_resid_post"
    compare(num_perturbations, perturb_location)

    perturb_location = "blocks.15.hook_resid_post"
    compare(num_perturbations, perturb_location)

    perturb_location = "blocks.30.hook_resid_post"
    compare(num_perturbations, perturb_location)

    perturb_location = "blocks.20.hook_resid_post"
    compare(num_perturbations, perturb_location)

    # num_perturbations = 3
    # perturb_location = "blocks.5.hook_resid_post"
    # compare(num_perturbations, perturb_location)

    # perturb_location = "blocks.15.hook_resid_post"
    # compare(num_perturbations, perturb_location)

    # perturb_location = "blocks.30.hook_resid_post"
    # compare(num_perturbations, perturb_location)

    # perturb_location = "blocks.20.hook_resid_post"
    # compare(num_perturbations, perturb_location)
