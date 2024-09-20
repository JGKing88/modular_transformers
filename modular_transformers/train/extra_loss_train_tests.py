import os

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
import accelerate
from accelerate import (Accelerator,
                        DeepSpeedPlugin)
from accelerate.utils import (LoggerType, DummyOptim, DummyScheduler)
from transformers import (AdamW,
                          AutoTokenizer,
                          AutoModelForCausalLM,
                          get_linear_schedule_with_warmup,
                         get_cosine_schedule_with_warmup,
                          set_seed,
                          AutoConfig)
from tqdm.auto import tqdm
import math
from modular_transformers.train.utils import Group_Texts
from pathlib import Path
import sys

from modular_transformers.models.gpt2.utils import initialize_gpt2_weights
import pickle

from modular_transformers.models.gpt2.configuration_gpt2 import GPT2Config
from modular_transformers.models import components

import matplotlib.pyplot as plt

import wandb

import random

import numpy as np

NUM_GPUS = 2
MAX_GPU_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32
CONTEXT_LENGTH = 1024

wandb.login(key="a338f755915cccd861b14f29bf68601d8e1ec2c9")

# Evaluate function
def evaluate(model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            batch = torch.stack(batch['input_ids']).transpose(1, 0)
            outputs = model(batch, labels=batch)
        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.stack(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    accelerator.print(f"validation loss: {loss.item()}, validation perplexity {perplexity.item()}")
    return loss.item(), perplexity.item()    


def main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier):
    #Set checkpoint if needed ------------------------------------------------
    chkpoint = None
    wandb_id = "amvcstc4"
    epoch_buffer = 12

    #Set training config --------------------------------------------

    data='100M'
    batch_size = 64

    train_config = {"lr": 0.0006, "num_epochs": 1, "correct_bias": True, "seed": 42, "batch_size": batch_size}
    tokenizer = AutoTokenizer.from_pretrained("gpt2", fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    path = '/om/weka/evlab/ehoseini/MyData/miniBERTa_v2/'
    grouped_pad_train = load_from_disk(
        os.path.join(path, f'miniBERTa-{data}-crunched',
                        f'train_context_len_{CONTEXT_LENGTH}'))
    grouped_pad_valid = load_from_disk(
        os.path.join(path, f'miniBERTa-{data}-crunched',
                        f'valid_context_len_{CONTEXT_LENGTH}'))

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if train_config['batch_size'] > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = train_config['batch_size'] // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE
        accelerator = Accelerator(log_with="wandb",gradient_accumulation_steps=gradient_accumulation_steps)
    else:
        accelerator = Accelerator(log_with="wandb")

    eval_dataloader = DataLoader(grouped_pad_valid, shuffle=False, batch_size=EVAL_BATCH_SIZE)
    train_dataloader = DataLoader(grouped_pad_train, shuffle=True, batch_size=batch_size)
    del grouped_pad_train, grouped_pad_valid

    #set model config ---------------------------------------

    config = {'regsize': 768, 'vocab_size': len(tokenizer), 'n_ctx': CONTEXT_LENGTH, 'bos_token_id': tokenizer.bos_token_id,
                        'eos_token_id': tokenizer.eos_token_id, "bottleneck": bottleneck_dim, "n_layer": n_layer, "loss_hooks": loss_hooks, "normalize_loss": normalize_loss,
                        "logit_multiplier": logit_multiplier, "inter_multiplier": inter_multiplier}
                        
    config = GPT2Config(config)
    model = components.LM(config)

    # model_name = ""
    # for layer in config.n_embds:
    #     model_name += f"{layer}-"
    model_name = f"{bottleneck_dim}x{n_layer}"

    if loss_hooks is not None:
        func_name = [name for name in loss_hooks.values()][0]
        model_name = f"{inter_multiplier}-{func_name}-{model_name}"
        run_name = f"multiplier_{inter_multiplier}"
    else:
        model_name = model_name
        run_name = "normal_loss"

    train_config.update(config.get())   

    project_name = "inner_reg_testing_test" 

    if chkpoint is not None:
        save_dir = Path(f'/om2/user/jackking/MyData/mt/miniberta_{data}/{model_name}/{run_name}/checkpoint_{chkpoint}')
        model = components.LM.from_pretrained(save_dir)
        accelerator.init_trackers(project_name, init_kwargs={"wandb": {"id": wandb_id, "resume": "must"}})
        api = wandb.Api()
        run = api.run(f"modular_transformers/curvature_testing/{wandb_id}")
        run.config["num_epochs"] = epoch_buffer + train_config["num_epochs"]
        run.update()
    else:
        accelerator.init_trackers(project_name, config=train_config, init_kwargs={"wandb": {"name": model_name}})

    torch.cuda.empty_cache()
    model.output_loss = True
    model_size = sum(t.numel() for t in model.parameters())
    print(f"{model_name} size: {model_size / 1000 ** 2:.1f}M parameters")
    # print(model)
    model = model.to(accelerator.device)

    # Define optimizer
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates AdamW Optimizer
    optimizer_cls = (torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(params=model.parameters(), lr=train_config['lr'])
    if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=(len(train_dataloader) * train_config['num_epochs']),
        )
    else:
        assert False

    # Pass everything to accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    # Logging variables
    n_steps_per_epoch = math.ceil(len(train_dataloader.dataset) / batch_size / NUM_GPUS)
    print(f"n_steps_per_epoch: {n_steps_per_epoch}")
    print(f"batch_size: {batch_size}")
    print(f"trainloader.dataset: {len(train_dataloader.dataset)}")
    print(f"trainloader: {len(train_dataloader)}")
    #should't num steps per epoch just be len(train_dataloader)?

    data_count = 0
    absolute_step = 0

    if loss_hooks is not None and normalize_loss:
        normalization_batch_size = 10
        extra_losses_list = np.zeros(normalization_batch_size)
        extra_losses_list[0] = 1
        logit_losses_list = np.zeros(normalization_batch_size)
        logit_losses_list[0] = 1
    
    count = 0

    for epoch in range(train_config['num_epochs']):
        model.train()
        torch.cuda.empty_cache()
        for step, batch in enumerate(train_dataloader):
            # if count == 5:
            #     break
            count += 1
            batch = [torch.stack(batch[x]).transpose(1, 0) for x in ['input_ids', 'attention_mask']]

            with accelerator.accumulate(model):
                outputs = model(batch[0], labels=batch[0], attention_mask=batch[1])
                logit_loss = outputs.loss

                if loss_hooks is not None:
                    extra_losses = model.output_extra_losses()
                    extra_loss = 0
                    # print("extra losses")
                    # print(extra_losses)
                    # print(extra_losses.values())
                    for loss_val in extra_losses.values():
                        if loss_val is not None:
                            extra_loss += loss_val
                    if extra_loss > 48:
                        print("HERE------------------")
                    # print(extra_loss)
                    # print(extra_loss.item())
                    
                    if normalize_loss and len(np.trim_zeros(logit_losses_list)) == normalization_batch_size:
                        logit_losses_list[step % normalization_batch_size] = logit_loss.item() * logit_multiplier
                        extra_losses_list[step % normalization_batch_size] = extra_loss.item() * inter_multiplier

                        logit_loss = logit_loss / np.mean(np.trim_zeros(logit_losses_list))
                        extra_loss = extra_loss / np.mean(np.trim_zeros(extra_losses_list))
                    
                    logit_loss = logit_loss * logit_multiplier
                    extra_loss = extra_loss * inter_multiplier
                    loss = logit_loss + extra_loss
                    logit_loss = logit_loss.item()
                    extra_loss = extra_loss.item()
                else:
                    loss = logit_loss
                    
                accelerator.backward(loss)
                lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

            data_count += batch[0].shape[0]

            absolute_step += 1

            accelerator.log({"train/train_loss": loss}, step=absolute_step)

            if loss_hooks is not None:
                accelerator.log({"train/inter_loss": extra_loss}, step=absolute_step)
                accelerator.log({"train/logit_loss": logit_loss}, step=absolute_step)

            accelerator.log({"train/epoch": (absolute_step + 1) / n_steps_per_epoch},
                            step=absolute_step)
            accelerator.log({"train/data_count": data_count}, step=absolute_step)
            accelerator.log({"train/learning_rate": lr_scheduler.get_last_lr()[0]}, step=absolute_step)
                
    accelerator.end_training()
    torch.cuda.empty_cache()

if __name__ == "__main__":

    bottleneck_dim = 48
    normalize_loss = False
    logit_multiplier = 1

    n_layer = 3

    # for func in ["", "l1_reg"]:
    #     for inter_multiplier in [1, 2, 5, 10]:
    #         #0 indexed
    #         loss_hooks = {2: func}
    #         main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier)
    
    # main(bottleneck_dim, n_layer, None, normalize_loss, 0, 0)
    
    for func in ["l0_curvature"]:
        for inter_multiplier in [1]:
            #0 indexed
            loss_hooks = {1: func}
            main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier)