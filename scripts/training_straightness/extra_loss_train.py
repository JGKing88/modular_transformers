import os

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
import accelerate
from accelerate import (Accelerator,
                        DeepSpeedPlugin)
from accelerate.utils import (LoggerType, DummyOptim, DummyScheduler)
from transformers import (AdamW,
                          AutoTokenizer,
                          GPT2LMHeadModel,
                          AutoModelForCausalLM,
                          get_linear_schedule_with_warmup,
                         get_cosine_schedule_with_warmup,
                          set_seed,
                          AutoConfig)
from transformers import GPT2LMHeadModel
from tqdm.auto import tqdm
import math
from modular_transformers.train.utils import Group_Texts
from pathlib import Path
import math

from modular_transformers.models.gpt2.utils import initialize_gpt2_weights
import pickle

from modular_transformers.models.gpt2.configuration_gpt2 import GPT2Config
from modular_transformers.models import components

import matplotlib.pyplot as plt

import wandb

import random

import numpy as np

NUM_GPUS = 1
MAX_GPU_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
CONTEXT_LENGTH = 1024

wandb.login(key="a338f755915cccd861b14f29bf68601d8e1ec2c9")

torch.manual_seed(0)
np.random.seed(0)
set_seed(0)

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


def main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier, pretrained, warmup, gradient_clipping, clip_loss, epochs, dropout_dict=None):
    #Set checkpoint if needed ------------------------------------------------
    chkpoint = None
    wandb_id = "amvcstc4"
    epoch_buffer = 12

    #Set training config --------------------------------------------

    data='100M'
    batch_size = 64

    lr = 0.0003
    train_config = {"lr": lr, "num_epochs": epochs, "correct_bias": True, "seed": 42, "batch_size": batch_size}
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
                        'eos_token_id': tokenizer.eos_token_id, "bottleneck": bottleneck_dim, "n_layer": n_layer, "loss_hooks": loss_hooks, "normalize_loss": normalize_loss, "dropout_dict": dropout_dict,
                        "logit_multiplier": logit_multiplier, "inter_multiplier": inter_multiplier, "n_heads": 12, "pretrained": pretrained, "warmup": warmup, "gradient_clipping": gradient_clipping, "clip_loss": clip_loss}
                        
    config = GPT2Config(config)
    model = components.LM(config)
    if pretrained:
        state_dict = torch.load('/om2/user/jackking/transformer_xray/scripts/dynamics/gpt2_weights.pt')
        model.load_state_dict(state_dict, strict=False)
    # model = GPT2LMHeadModel.from_pretrained("gpt2")
    # config = GPT2Config()
    # model = GPT2LMHeadModel(config)

    # model_name = ""
    # for layer in config.n_embds:
    #     model_name += f"{layer}-"
    num_epochs = train_config["num_epochs"]
    model_name = f"epochs{num_epochs}-{bottleneck_dim}x{n_layer}"

    if loss_hooks is not None:
        func_name = [(layer, name) for layer, name in loss_hooks.items()][0]
        model_name = f"lr{lr}-{inter_multiplier}-layer{func_name[0]}-{func_name[1]}-{model_name}"       
        if normalize_loss:
            model_name = f"scaled-{model_name}"
        if warmup > 0:
            model_name = f"warmup{warmup}-{model_name}"
        if len(loss_hooks) > 1:
            model_name = f"multi-{model_name}"
        if clip_loss:
            model_name = f"clip-{model_name}"
        # model_name = f"fastbeforeattn-{model_name}"
    
    if dropout_dict is not None:
        dropout_str = ""
        for layer, location in dropout_dict.items():
            for location, dropout in location.items():
                dropout_str += f"{layer}-{location}-{dropout}-"
        model_name = f"{dropout_str}{model_name}"
    
    if pretrained:
        model_name = f"finetuned-{model_name}"  

    # model_name = "gpt2_trainedmore"
    
    # model_name = f"huggingface-pretrained-{model_name}"

    # train_config.update(config)
    train_config.update(config.get())   

    project_name = "inner_reg_testing" 

    if chkpoint is not None:
        save_dir = Path(f'/om2/user/jackking/MyData/mt/miniberta_{data}/{model_name}/checkpoint_{chkpoint}')
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
    print("Device:", accelerator.device)    

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
            num_training_steps=math.ceil(len(train_dataloader) * train_config['num_epochs'] / gradient_accumulation_steps),
        )
    else:
        assert False

    # Pass everything to accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    # Logging variables
    # n_steps_per_epoch = math.ceil(len(train_dataloader.dataset) / batch_size / NUM_GPUS)
    n_steps_per_epoch = len(train_dataloader)
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

    orig_multiplier = inter_multiplier

    max_grad_norm = gradient_clipping
    prev_loss = 0

    num_warmup_steps = len(train_dataloader) * warmup
    print(f"num_warmup_steps: {num_warmup_steps}")

    if loss_hooks is not None:
        model.remove_hooks()
    model.eval()
    with torch.no_grad():
        valid_loss, valid_accuracy = evaluate(model, eval_dataloader, accelerator)
    accelerator.log({"validation/valid_loss": valid_loss}, step=absolute_step)
    accelerator.log({"validation/valid_accuracy": valid_accuracy}, step=absolute_step)
    if loss_hooks is not None:
        model.set_hooks()
    torch.cuda.empty_cache()

    for epoch in tqdm(range(train_config["num_epochs"])):

        model.train()
        torch.cuda.empty_cache()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            batch = [torch.stack(batch[x]).transpose(1, 0) for x in ['input_ids', 'attention_mask']]

            if loss_hooks is None:
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    outputs = model(batch[0], labels=batch[0], attention_mask=batch[1])
                    loss = outputs.loss 
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if gradient_clipping is not None:
                            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                    gradients = []
                    for param in model.parameters():
                        gradients.append(param.grad)
                    gradient_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients if g is not None]))

                    lr_scheduler.step()
                    optimizer.step()
                    
                torch.cuda.empty_cache()
            else:
                with accelerator.accumulate(model):
                    optimizer.zero_grad() 

                    outputs = model.forward_with_extra_loss(batch[0], labels=batch[0], attention_mask=batch[1], attn_indices = (batch[1] == 0).nonzero(as_tuple=False).t().detach().cpu())
                    logit_loss = outputs.loss

                    # if torch.isnan(logit_loss):
                    #     accelerator.log({"none_loss": 1}, step=absolute_step)
                    #     optimizer.zero_grad()
                    #     torch.cuda.empty_cache()
                    #     continue

                    extra_losses = model.output_extra_losses()
                    extra_loss = 0
                    for loss in extra_losses.values():
                        if loss is not None:
                            extra_loss += loss
                    
                    # if normalize_loss and len(np.trim_zeros(logit_losses_list)) == normalization_batch_size:
                    #     logit_losses_list[step % normalization_batch_size] = logit_loss.item() * logit_multiplier
                    #     extra_losses_list[step % normalization_batch_size] = extra_loss.item() * inter_multiplier

                    #     logit_loss = logit_loss / np.mean(np.trim_zeros(logit_losses_list))
                    #     extra_loss = extra_loss / np.mean(np.trim_zeros(extra_losses_list))
                            
                    if normalize_loss:
                        # inter_multiplier = logit_loss.item() / extra_loss.item()
                        inter_multiplier = orig_multiplier * (logit_loss.item() / extra_loss.item())
                        # inter_multiplier = orig_multiplier * (logit_loss / extra_loss) #previous "scaled" one
                    
                    try:
                        warmup_multiplier = 1 / (1 + math.e**(-(8 / (num_warmup_steps - math.ceil(np.log10(abs(inter_multiplier)) + 1)) * (absolute_step - num_warmup_steps/2))))
                    except:
                        warmup_multiplier = 1

                    base_logit_loss = logit_loss.item()
                    base_extra_loss = extra_loss.item()
                        
                    logit_loss = logit_loss * logit_multiplier
                    extra_loss = extra_loss * inter_multiplier * warmup_multiplier

                    if clip_loss:
                        if extra_loss - prev_loss > 0.001 and prev_loss != 0:
                            extra_loss = extra_loss * (prev_loss / extra_loss.item()) + 0.001
                        prev_loss = extra_loss.item()

                    loss = logit_loss + extra_loss
                        
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if gradient_clipping is not None:
                            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                    lr_scheduler.step()
                    optimizer.step()

                    gradients = []
                    for param in model.parameters():
                        gradients.append(param.grad)
                    gradient_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients if g is not None]))
                
                accelerator.log({"train/inter_loss": base_extra_loss}, step=absolute_step)
                accelerator.log({"train/logit_loss": base_logit_loss}, step=absolute_step)
                accelerator.log({"train/warmup_multiplier": warmup_multiplier}, step=absolute_step)

            data_count += batch[0].shape[0]

            absolute_step += 1

            accelerator.log({"train/train_loss": loss}, step=absolute_step)
            accelerator.log({"train/gradient": gradient_norm}, step=absolute_step)

            torch.cuda.empty_cache()

            accelerator.log({"train/epoch": (absolute_step + 1) / n_steps_per_epoch},
                            step=absolute_step)
            accelerator.log({"train/data_count": data_count}, step=absolute_step)
            accelerator.log({"train/learning_rate": lr_scheduler.get_last_lr()[0]}, step=absolute_step)

            if absolute_step % 200 == 0:
                torch.cuda.empty_cache()
                if loss_hooks is not None:
                    model.remove_hooks()
                model.eval()
                with torch.no_grad():
                    valid_loss, valid_accuracy = evaluate(model, eval_dataloader, accelerator)
                accelerator.log({"validation/valid_loss": valid_loss}, step=absolute_step)
                accelerator.log({"validation/valid_accuracy": valid_accuracy}, step=absolute_step)
                if loss_hooks is not None:
                    model.set_hooks()
                model.train()
                torch.cuda.empty_cache()
            
            if absolute_step % 20000 == 0:
                save_dir = Path(
                    f'/om2/user/jackking/MyData/mt/miniberta_{data}/{model_name}/checkpoint_{absolute_step}')
                save_dir.mkdir(parents=True, exist_ok=True)
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(save_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save,state_dict=accelerator.get_state_dict(model))
                accelerator.save(
                    {
                        "epoch": epoch,"steps": step,"optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                    },os.path.join(save_dir,'accelerator_states'))

    save_dir = Path(
        f'/om2/user/jackking/MyData/mt/miniberta_{data}/{model_name}/checkpoint_final')
    save_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save,state_dict=accelerator.get_state_dict(model))
    accelerator.save(
        {
            "epoch": epoch,"steps": step,"optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
        },os.path.join(save_dir,'accelerator_states'))
                
    accelerator.end_training()
    torch.cuda.empty_cache()

if __name__ == "__main__":

    bottleneck_dim = 768
    normalize_loss = False
    logit_multiplier = 1
    inter_multiplier = 0.01
    gradient_clipping = 1
    clip_loss = False

    n_layer = 12
    pretrained = False
    warmup = 10
    epochs = 10


    loss_hooks = None

    # layer = 7
    # dropout = 0.8
    # dropout_dict = {layer: {"attn": dropout, "embd": dropout}}
    # main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier, pretrained, warmup, gradient_clipping, clip_loss, epochs, dropout_dict=dropout_dict)
    # dropout = 0.5
    # dropout_dict = {layer: {"attn": dropout, "embd": dropout}}
    # main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier, pretrained, warmup, gradient_clipping, clip_loss, epochs, dropout_dict=dropout_dict)

    # layer = 3
    # dropout = 0.5
    # dropout_dict = {layer: {"attn": dropout, "embd": dropout}}
    # main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier, pretrained, warmup, gradient_clipping, clip_loss, epochs, dropout_dict=dropout_dict)
    # dropout = 0.8
    # dropout_dict = {layer: {"attn": dropout, "embd": dropout}}
    # main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier, pretrained, warmup, gradient_clipping, clip_loss, epochs, dropout_dict=dropout_dict)
    

    # layer = 3
    # dropout = 0.8
    # dropout_dict = {layer: {section: dropout}}
    # main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier, pretrained, warmup, gradient_clipping, clip_loss, epochs, dropout_dict=dropout_dict)

    # layer = 3
    # dropout = 0.5
    # dropout_dict = {layer: {section: dropout}}
    # main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier, pretrained, warmup, gradient_clipping, clip_loss, epochs, dropout_dict=dropout_dict)

    loss_hooks = {7: "sparsity"}
    inter_multiplier = 0.001
    dropout_dict = None
    main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier, pretrained, warmup, gradient_clipping, clip_loss, epochs, dropout_dict=dropout_dict)

    loss_hooks = {7: "l2_curvature"}
    main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier, pretrained, warmup, gradient_clipping, clip_loss, epochs, dropout_dict=dropout_dict)


        # dropout_dict[layer] = {"attn": dropout, "embd": dropout}
        # main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier, pretrained, warmup, gradient_clipping, clip_loss, epochs, dropout_dict)

    # for func in ["", "l1_reg"]:
    #     for inter_multiplier in [1, 2, 5, 10]:
    #         #0 indexed
    #         loss_hooks = {2: func}
    #         main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier)
    
    # main(bottleneck_dim, n_layer, None, normalize_loss, 0, 0)
    # loss_hooks = None
    # main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier, pretrained, warmup, gradient_clipping, clip_loss)
    

    # main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, inter_multiplier, pretrained, warmup, gradient_clipping, clip_loss, epochs, dropout_dict)

    # loss_hooks = {4: "l1_curvature", 5: "l1_curvature", 6: "l1_curvature"}
    # main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, 1, pretrained, lag)

    # loss_hooks = {4: "l2_curvature", 5: "l2_curvature", 6: "l2_curvature"}
    # main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, 1, pretrained, lag)

    # loss_hooks = {4: "l0_curvature", 5: "l0_curvature", 6: "l0_curvature"}
    # main(bottleneck_dim, n_layer, loss_hooks, normalize_loss, logit_multiplier, 1, pretrained, lag)
