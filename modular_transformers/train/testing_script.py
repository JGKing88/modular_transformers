import os

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import LoggerType, DummyOptim, DummyScheduler
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    set_seed,
    AutoConfig,
)
from tqdm.auto import tqdm
import math
from modular_transformers.train.utils import Group_Texts
from pathlib import Path
import sys

from modular_transformers.models.gpt2.utils import initialize_gpt2_weights
import pickle

from modular_transformers.models.gpt2.configuration_gpt2 import GPT2Config
from modular_transformers.models import components

import wandb

CONTEXT_LENGTH = 1024
MAX_GPU_BATCH_SIZE = 16

if __name__ == "__main__":
    data = "100M"
    batch_size = 64

    # Set training config --------------------------------------------

    train_config = {
        "lr": 0.0006,
        "num_epochs": 12,
        "correct_bias": True,
        "seed": 42,
        "batch_size": batch_size,
    }
    tokenizer = AutoTokenizer.from_pretrained("gpt2", fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    path = "/om/weka/evlab/ehoseini/MyData/miniBERTa_v2/"
    grouped_pad_train = load_from_disk(
        os.path.join(
            path, f"miniBERTa-{data}-crunched", f"train_context_len_{CONTEXT_LENGTH}"
        )
    )
    grouped_pad_valid = load_from_disk(
        os.path.join(
            path, f"miniBERTa-{data}-crunched", f"valid_context_len_{CONTEXT_LENGTH}"
        )
    )

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 4
    batch_size = MAX_GPU_BATCH_SIZE
    accelerator = Accelerator(
        log_with="wandb", gradient_accumulation_steps=gradient_accumulation_steps
    )

    eval_dataloader = DataLoader(grouped_pad_valid, shuffle=False, batch_size=16)
    train_dataloader = DataLoader(
        grouped_pad_train, shuffle=True, batch_size=batch_size
    )
    del grouped_pad_train, grouped_pad_valid

    # set model config ---------------------------------------
    bottleneck_dim = 128
    n_layer = 5

    config = {
        "vocab_size": len(tokenizer),
        "n_ctx": CONTEXT_LENGTH,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bottleneck": bottleneck_dim,
        "n_layer": n_layer,
    }
    config = GPT2Config(config)
    model = components.LM(config)

    torch.cuda.empty_cache()
    model.output_loss = True
    model_size = sum(t.numel() for t in model.parameters())
    # print(f"{model_name} size: {model_size / 1000 ** 2:.1f}M parameters")
    # print(model)
    model = model.to(accelerator.device)

    # Define optimizer
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates AdamW Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(params=model.parameters(), lr=train_config["lr"])
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=(len(train_dataloader) * train_config["num_epochs"])
            // gradient_accumulation_steps,
        )
    else:
        assert False

    # Pass everything to accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    # Logging variables
    n_steps_per_epoch = math.ceil(
        len(train_dataloader.dataset) / train_config["batch_size"]
    )
    print(n_steps_per_epoch)
    n_steps_per_epoch = math.ceil(len(train_dataloader.dataset) / 16)
    print(n_steps_per_epoch)
    n_steps_per_epoch = len(train_dataloader.dataset)
    print(n_steps_per_epoch)

    # set_seed(50)

    # tokenizer = AutoTokenizer.from_pretrained("gpt2", fast=False)
    # tokenizer.pad_token = tokenizer.eos_token

    # override_vars = {'vocab_size': len(tokenizer), 'n_ctx': CONTEXT_LENGTH, 'bos_token_id': tokenizer.bos_token_id,
    #                  'eos_token_id': tokenizer.eos_token_id, "bottleneck": 576, "n_layer": 7}
    # config = GPT2Config(override_vars)
    # print(config.get()["n_embds"])
    # model = components.LM(config)

    # model_size = sum(t.numel() for t in model.parameters())
    # print(f"Model size: {model_size / 1000 ** 2:.1f}M parameters")

    # state_dict = model.state_dict()
    # print(state_dict['transformer.h.1.mlp.c_proj.weight'])
