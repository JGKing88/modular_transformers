import torch
import math

from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed, AutoModel, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from transformers import BatchEncoding
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from minicons import scorer
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_convert

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
import textwrap


#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#set tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

#poor loss:  "warmup2-finetuned-lr0.0006-100-l2_curvature-768x12"
fine_tuned_models = ["finetuned-warmup2-lr0.0003-100-layer9-l2_curvature-768x12", "finetuned-768x12", "warmup2-finetuned-lr0.0003-100-l2_curvature-768x12", "multi-warmup2-finetuned-lr0.0003-1-l2_curvature-768x12", "gpt2", "finetuned-epochs20-768x12", "finetuned-warmup2-lr0.0002-1-layer9-l2_curvature-epochs20-768x12", "finetuned-warmup2-lr0.0003-100-layer8-l2_curvature-epochs10-768x12", "finetuned-warmup2-lr0.0006--1-layer8-l2_curvature-epochs10-768x12"]
fulltrained_models = ["warmup10-lr0.0003-1-layer9-l2_curvature-768x12", "warmup10-lr0.0003-1-l2_curvature-768x12", "warmup5-lr0.0006-1-l2_curvature-768x12", '768x12', "warmup10-lr0.0006-1-layer9-l2_curvature-epochs20-768x12", "warmup5-lr0.0006-1-layer8-l2_curvature-epochs10-768x12",  "warmup20-lr0.0006-1-layer8-l2_curvature-epochs20-768x12", "warmup10-lr0.0006-1-layer8-l2_curvature-epochs10-768x12", "epochs20-768x12", "warmup10-lr0.0003-1-layer8-l2_curvature-epochs10-768x12", "warmup10-lr0.0006-1-layer8-l1_curvature-epochs10-768x12", "warmup10-lr0.0006-1-layer9-l2_curvature-epochs10-768x12", "warmup10-lr0.0006-0.1-layer8-l2_curvature-epochs10-768x12", "warmup10-lr0.0006--0.1-layer8-l2_curvature-epochs10-768x12", "warmup10-lr0.0006-1-layer8-l0_curvature-epochs10-768x12"]

dir_path = "/om2/user/jackking/modular_transformers/scripts/training_straightness/data"

# model_names = fine_tuned_models + fulltrained_models

# model_names = ["beforeattn-warmup10-lr0.0003-1-layer8-l2_curvature-epochs10-768x12", "warmup10-lr0.0003-1-layer8-l2_curvature-epochs10-768x12", "warmup5-lr0.0006-1-l2_curvature-768x12", "768x12"]

# model_names = ["fastbeforeattn-warmup2-lr0.002-1-layer8-l2_curvature-epochs3-768x12"]

model_names = ["7-embd-0.5-epochs10-768x12", "768x12", "warmup10-lr0.0003-1-layer7-l2_curvature-epochs10-768x12", "7-attn-0.8-7-embd-0.8-epochs10-768x12", "warmup10-lr0.0003-0.01-layer7-sparsity-epochs10-768x12", "3-attn-0.5-epochs10-768x12",
               "3-attn-0.8-epochs10-768x12", "3-embd-0.8-epochs10-768x12", "3-embd-0.5-epochs10-768x12", "7-embd-0.8-epochs10-768x12", "7-attn-0.8-epochs10-768x12", "7-attn-0.8-epochs10-768x12", "warmup20-lr0.0003-0.1-layer7-l0_curvature-epochs10-768x12", "warmup10-lr0.0003-0.1-layer7-l0_curvature-epochs10-768x12"]

model_names = ["gpt2", "7-embd-0.5-epochs10-768x12", "768x12", "warmup10-lr0.0003-1-layer7-l2_curvature-epochs10-768x12", "warmup10-lr0.0003-0.01-layer7-sparsity-epochs10-768x12"]

def generate_random_pertubation(shape, pertubation_size):
    pertubation = torch.randn(shape)
    pertubation = pertubation / np.linalg.norm(pertubation) * pertubation_size
    return pertubation

def random_perturbation_function(perturbation_size):
    def random_perturbation_function_inner(input, layer, token):
        size = np.linalg.norm(input.cpu().detach().numpy()).item() * perturbation_size
        return generate_random_pertubation(input.shape, size)
    return random_perturbation_function_inner

def compute_surprisals(model_names, dataloader, perturbation_type, surprisal_type, perturbation_size = None, perturbation_layer = None, perturbation_section=None):
    try:
        surprisal_dict = torch.load(f"{dir_path}/surprisal_dict.pt", map_location=torch.device('cpu'))
    except:
        surprisal_dict = {}

    if "activation" in perturbation_type:
        perturbation_type = f"activation_random_size{perturbation_size}_layer{perturbation_layer}_{perturbation_section}"
    if perturbation_type not in surprisal_dict:
        surprisal_dict[perturbation_type] = {}
    if surprisal_type not in surprisal_dict[perturbation_type]:
        surprisal_dict[perturbation_type][surprisal_type] = {}

    for model_name in model_names:
        print(model_name)
        if model_name in surprisal_dict[perturbation_type][surprisal_type]:
            continue
        if model_name == "gpt2":
            model = GPT2LMHeadModel.from_pretrained('gpt2')
        else:
            path = f'/om2/user/jackking/MyData/mt/miniberta_100M/{model_name}/checkpoint_final'
            # path = f'/om2/user/jackking/MyData/mt/miniberta_100M/{model_name}/checkpoint_20000'
            model = AutoModel.from_pretrained(path)
            state_dict = model.state_dict()
            config = AutoConfig.from_pretrained('gpt2')
            model = GPT2LMHeadModel(config)
            model.load_state_dict(state_dict, strict = False)
        
        if "activation" in perturbation_type:
            perturbation_hooks = {perturbation_layer: [(perturbation_section, "all", random_perturbation_function(perturbation_size))]}
            register_pertubation_hooks(model, perturbation_hooks, device)
            
        model = scorer.IncrementalLMScorer(model, tokenizer=tokenizer, device=device)
        all_surprisals = []
        for batch in tqdm(dataloader):
            new_batch = {}
            new_batch["input_ids"] = torch.stack(batch["input_ids"]).transpose(1, 0)
            new_batch["attention_mask"] = torch.stack(batch["attention_mask"]).transpose(1, 0)
            new_batch = BatchEncoding(new_batch)
            surprisals = model.sequence_score(new_batch, reduction = lambda x: -x.sum(0))
            all_surprisals.extend(surprisals)
        surprisal_dict[perturbation_type][surprisal_type][model_name] = np.array(all_surprisals)

    with open(f"{dir_path}/surprisal_dict.pt", 'wb') as f:
        torch.save(surprisal_dict, f)


def compute_surprisals_with_5000_sentences(model_names, perturbation_type, surprisal_type, perturbation_size = None, perturbation_layer = None, perturbation_section=None):
    try:
        surprisal_dict = torch.load(f"{dir_path}/surprisal_dict.pt", map_location=torch.device('cpu'))
    except:
        surprisal_dict = {}

    data_dir = "/rdma/vast-rdma/vast/evlab/ehoseini/MyData/sent_sampling/analysis/straightening/generation/sentences_ud_sentencez_token_filter_v3_textNoPeriod_cntx_3_cont_7.pkl"
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
    tokenizer.pad_token = tokenizer.eos_token
    data = tokenizer.batch_encode_plus(data, add_special_tokens=True, padding='longest', return_tensors="pt")["input_ids"]
    data = data[:5000]
    data = data[:, :10].to(device)

    if "activation" in perturbation_type:
        perturbation_type = f"activation_random_size{perturbation_size}_layer{perturbation_layer}_{perturbation_section}"
    if perturbation_type not in surprisal_dict:
        surprisal_dict[perturbation_type] = {}
    if surprisal_type not in surprisal_dict[perturbation_type]:
        surprisal_dict[perturbation_type][surprisal_type] = {}

    for model_name in model_names:
        print(model_name)
        # if model_name in surprisal_dict[perturbation_type][surprisal_type]:
        #     continue
        if model_name == "gpt2":
            model = GPT2LMHeadModel.from_pretrained('gpt2')
        else:
            path = f'/om2/user/jackking/MyData/mt/miniberta_100M/{model_name}/checkpoint_final'
            # path = f'/om2/user/jackking/MyData/mt/miniberta_100M/{model_name}/checkpoint_20000'
            model = AutoModel.from_pretrained(path)
            # state_dict = model.state_dict()
            # config = AutoConfig.from_pretrained('gpt2')
            # model = GPT2LMHeadModel(config)
            # model.load_state_dict(state_dict, strict = False)

        model.to(device)
        
        if "activation" in perturbation_type:
            perturbation_hooks = {perturbation_layer: [(perturbation_section, "all", random_perturbation_function(perturbation_size))]}
            register_pertubation_hooks(model, perturbation_hooks, device)
            

        # model = scorer.IncrementalLMScorer(model, tokenizer=tokenizer, device=device)
        # all_surprisals = []
        # for i in range(0, 5000, 1000):
        #     all_surprisals.extend(torch.tensor(model.sequence_score(data[i:i+1000], reduction = lambda x: -x.sum(0))))

        # judge_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True).to(device)
        # surprisals = []
        # for i in range(0, 5000, 100):
        #     input = data[i:i+100]
        #     logits = model(input)[0]
        #     #batch_size x seq_len
        #     output_tokens = np.argmax(logits.cpu().detach().numpy(), axis=-1)
        #     print(tokenizer.decode(input[0]))
        #     print(tokenizer.decode(output_tokens[0]))
        #     print()

        #     judge_logits = judge_model(input)[0]
        #     judge_log_probs = F.log_softmax(judge_logits, dim=-1).cpu().detach().numpy()

        #     for i in range(input.shape[0]):
        #         #for each position in the sentence, find the surprisal of the token in the next position in the input sentence
        #         token_log_probs = -judge_log_probs[i, np.arange(len(output_tokens[i])), output_tokens[i]]
        #         surprisals.extend(token_log_probs)

        surprisals = []
        for i in range(0, 5000, 100):
            input = data[i:i+100]
            logits = model(input)[0]
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            for i in range(input.shape[0]):
                surprisals.append(-1 * torch.sum(probs[i] * log_probs[i], dim=-1).detach().cpu().numpy())
    
        surprisal_dict[perturbation_type][surprisal_type][model_name] = np.array(surprisals)

    with open(f"{dir_path}/surprisal_dict.pt", 'wb') as f:
        torch.save(surprisal_dict, f)


def compute_surprisals_with_context(model_names, prefixes, queries, perturbation_type, surprisal_type, perturbation_size=None, perturbation_layer=None, perturbation_section=None):
    try:
        surprisal_dict = torch.load(f"{dir_path}/surprisal_dict.pt", map_location=torch.device('cpu'))
    except:
        surprisal_dict = {}

    if "activation" in perturbation_type:
        perturbation_type = f"activation_random_size{perturbation_size}_layer{perturbation_layer}_{perturbation_section}"
    if perturbation_type not in surprisal_dict:
        surprisal_dict[perturbation_type] = {}
    if surprisal_type not in surprisal_dict[perturbation_type]:
        surprisal_dict[perturbation_type][surprisal_type] = {}

    for model_name in tqdm(model_names):
        print(model_name)
        if model_name in surprisal_dict[perturbation_type][surprisal_type]:
            continue
        if model_name == "gpt2":
            model = GPT2LMHeadModel.from_pretrained('gpt2')
        else:
            path = f'/om2/user/jackking/MyData/mt/miniberta_100M/{model_name}/checkpoint_final'
            # path = f'/om2/user/jackking/MyData/mt/miniberta_100M/{model_name}/checkpoint_20000'
            model = AutoModel.from_pretrained(path)
            state_dict = model.state_dict()
            config = AutoConfig.from_pretrained('gpt2')
            model = GPT2LMHeadModel(config)
            model.load_state_dict(state_dict, strict = False)

        if "activation" in perturbation_type:
            perturbation_hooks = {perturbation_layer: [(perturbation_section, "all", random_perturbation_function(perturbation_size))]}
            register_pertubation_hooks(model, perturbation_hooks, device)

        model = scorer.IncrementalLMScorer(model, tokenizer=tokenizer, device=device)
        all_surprisals = []
        for prefix, query in tqdm(zip(prefixes, queries)):
            surprisals = model.conditional_score(prefix, query, reduction = lambda x: -x.sum(0))
            all_surprisals.extend(surprisals)
        surprisal_dict[perturbation_type][surprisal_type][model_name] = np.array(all_surprisals)

    with open(f"{dir_path}/surprisal_dict.pt", 'wb') as f:
        torch.save(surprisal_dict, f)


def perturb_inputs(input_ids, perturbation_type):
    perturbation_amount = math.ceil(len(input_ids) * 0.1)
    if perturbation_type == "swap":
        for i in range(perturbation_amount):
            idx1, idx2 = np.random.choice(len(input_ids), 2)
            input_ids[idx1], input_ids[idx2] = input_ids[idx2], input_ids[idx1]
        return input_ids
    elif perturbation_type == "remove":
        for i in range(perturbation_amount):
            idx = np.random.choice(len(input_ids))
            input_ids[idx] = tokenizer.pad_token_id
        return input_ids
    elif perturbation_type == "replace":
        for i in range(perturbation_amount):
            idx = np.random.choice(len(input_ids))
            input_ids[idx] = np.random.choice(len(tokenizer))
        return input_ids
    else:
        raise ValueError(f"perturbation_type {perturbation_type} not recognized")
    

def get_cont_qp(data, perturbation_type):
    queries = [tokenizer.decode(sample[20:40]) for sample in data["input_ids"][:500]]
    if perturbation_type == "none" or "activation" in perturbation_type:
        prefixes = [tokenizer.decode(sample[:20]) for sample in data["input_ids"][:500]]
    else:
        prefixes = [sample[:20] for sample in data["input_ids"][:500]]
        prefixes = [tokenizer.decode(perturb_inputs(prefix, perturbation_type)) for prefix in prefixes]
    
    return queries, prefixes

def get_qa_qp(dataset, perturbation_type):
    queries = dataset["best_answer"]
    if perturbation_type == "none" or "activation" in perturbation_type:
        prefixes = dataset["question"]
    else:
        prefixes = [tokenizer.encode(question) for question in dataset["question"]]
        prefixes = [tokenizer.decode(perturb_inputs(prefix, perturbation_type)) for prefix in prefixes]
    
    return queries, prefixes

def get_math_qp(dataset, perturbation_type):
    queries = [sample["answer"] for sample in dataset]
    if perturbation_type == "none" or "activation" in perturbation_type:
        prefixes = [sample["question"] for sample in dataset]
    else:
        prefixes = [tokenizer.encode(sample["question"]) for sample in dataset]
        prefixes = [tokenizer.decode(perturb_inputs(prefix, perturbation_type)) for prefix in prefixes]
    
    return queries, prefixes

if __name__ == "__main__":

    surprisal_type = "5000sentences.new"
    compute_surprisals_with_5000_sentences(model_names, "none", surprisal_type)

    # perturbation_type = "activation"
    # for perturbation_layer in [3, 7]:#[8, 1, 4, 5, 9]:
    #     for perturbation_section in ["post_block"]: #[""mid_block", "before_mlp", "before_attn"]:
    #         for perturbation_size in [0.1, 0.5, 5, 1, 2]:
    #             print(f"here2: {perturbation_layer}, {perturbation_section}, {perturbation_size}")
    #             compute_surprisals_with_5000_sentences(model_names, perturbation_type, surprisal_type, perturbation_size, perturbation_layer, perturbation_section)
    
                
    # surprisal_type = "regular.new"
    # path = '/om/weka/evlab/ehoseini/MyData/miniBERTa_v2/'
    # data_size = "10M"
    # data = load_from_disk(
    #     os.path.join(path, f'miniBERTa-{data_size}-crunched',
    #                     f'valid_context_len_{1024}'))

    # #NOTE: scorer takes into account the attention mask in the dataloader, so don't need to worry about that
    # dataloader = DataLoader(data, shuffle=False, batch_size=64)

    # compute_surprisals(model_names, dataloader, "none", surprisal_type)
    # perturbation_type = "activation"
    # for perturbation_layer in [3, 7]:#[8, 1, 4, 5, 9]:
    #     for perturbation_section in ["post_block"]: #[""mid_block", "before_mlp", "before_attn"]:
    #         for perturbation_size in [0.1, 0.5, 5, 1, 2]:
    #             compute_surprisals(model_names, dataloader, perturbation_type, surprisal_type, perturbation_size, perturbation_layer, perturbation_section)
                

    # surprisal_type = "continuation"
    # path = '/om/weka/evlab/ehoseini/MyData/miniBERTa_v2/'
    # data_size = "10M"
    # data = load_from_disk(
    #     os.path.join(path, f'miniBERTa-{data_size}-crunched',
    #                     f'valid_context_len_{512}'))
    
    # for perturbation_type in ["none", "swap", "remove", "replace"]:
    #     queries, prefixes = get_cont_qp(data, perturbation_type)
    #     compute_surprisals_with_context(model_names, prefixes, queries, perturbation_type, surprisal_type)

    # queries, prefixes = get_cont_qp(data, "none")
    # perturbation_type = "activation"
    # for perturbation_layer in [3, 7]:#[8, 1, 4, 5, 9]:
    #     for perturbation_section in ["post_block"]: #[""mid_block", "before_mlp", "before_attn"]:
    #         for perturbation_size in [0.1, 0.5, 5, 1, 2]:
    #             compute_surprisals_with_context(model_names, prefixes, queries, perturbation_type, surprisal_type, perturbation_size, perturbation_layer, perturbation_section)



    # surprisal_type = "question_answering"
    # dataset = load_dataset('truthful_qa', 'generation', split='validation')

    # # for perturbation_type in ["none", "swap", "remove", "replace"]:
    # #     queries, prefixes = get_qa_qp(dataset, perturbation_type)
    # #     compute_surprisals_with_context(model_names, prefixes, queries, perturbation_type, surprisal_type)

    # perturbation_type = "activation"
    # for perturbation_layer in [3, 7]:#[8, 1, 4, 5, 9]:
    #     for perturbation_section in ["post_block"]: #[""mid_block", "before_mlp", "before_attn"]:
    #         for perturbation_size in [0.1, 0.5, 5, 1, 2]:
    #             compute_surprisals_with_context(model_names, prefixes, queries, perturbation_type, surprisal_type, perturbation_size, perturbation_layer, perturbation_section)



    # surprisal_type = "math"
    # dataset = list(open(f"{dir_path}/grade_school_math.jsonl"))
    # dataset = [json.loads(sample) for sample in dataset]

    # # for perturbation_type in ["none", "swap", "remove", "replace"]:
    # #     queries, prefixes = get_math_qp(dataset, perturbation_type)
    # #     compute_surprisals_with_context(model_names, prefixes, queries, perturbation_type, surprisal_type)

    # perturbation_type = "activation"
    # for perturbation_layer in [3, 7]:#[8, 1, 4, 5, 9]:
    #     for perturbation_section in ["post_block"]: #[""mid_block", "before_mlp", "before_attn"]:
    #         for perturbation_size in [0.1, 0.5, 5, 1, 2]:
    #             compute_surprisals_with_context(model_names, prefixes, queries, perturbation_type, surprisal_type, perturbation_size, perturbation_layer, perturbation_section)
