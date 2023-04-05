import os

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
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

"""
Basic script to train a distill-gpt2 model using accelerate and grouping function.
Set config to use DeepSpeed
'accelerate config' -> enter in desired DeepSpeed configs or input path to deepspeed_config.json
'accelerate launch bplm/basic_accelerate_addedAug2022.py'
"""

MAX_GPU_BATCH_SIZE = 8
EVAL_BATCH_SIZE =32
CONTEXT_LENGTH = 1024

# Evaluate function
def evaluate():
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

if __name__ == "__main__":
    run_name='gaussian_init_1'
    model_name='gpt2'
    data='10M'
    #chkpoint=43750
    chkpoint = None
    train_config = {"lr": 0.0006, "num_epochs": 100, "correct_bias": True, "seed": 42, "batch_size": 64}
    tokenizer = AutoTokenizer.from_pretrained("gpt2", fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    grouped_pad_train = load_from_disk(
        os.path.join('/om/user/ehoseini/MyData/miniBERTa_v2/', f'miniBERTa-{data}-crunched',
                     f'train_context_len_{CONTEXT_LENGTH}'))
    grouped_pad_test = load_from_disk(
        os.path.join('/om/user/ehoseini/MyData/miniBERTa_v2/', f'miniBERTa-{data}-crunched',
                     f'test_context_len_{CONTEXT_LENGTH}'))
    grouped_pad_valid = load_from_disk(
        os.path.join('/om/user/ehoseini/MyData/miniBERTa_v2/', f'miniBERTa-{data}-crunched',
                     f'valid_context_len_{CONTEXT_LENGTH}'))
    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 8
    if train_config['batch_size'] > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = train_config['batch_size'] // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE
    accelerator = Accelerator(log_with="wandb",gradient_accumulation_steps=gradient_accumulation_steps)

    eval_dataloader = DataLoader(grouped_pad_valid, shuffle=False, batch_size=EVAL_BATCH_SIZE)
    test_dataloader = DataLoader(grouped_pad_test, shuffle=True, batch_size=EVAL_BATCH_SIZE)
    train_dataloader = DataLoader(grouped_pad_train, shuffle=True, batch_size=batch_size)
    del grouped_pad_train, grouped_pad_valid, grouped_pad_test

    # Logging initialization
    # Change name test to log to different project
    accelerator.init_trackers("bplm_gpt2", config=train_config,init_kwargs={'name':run_name})

    config = GPT2Config(vocab_size=len(tokenizer), n_ctx=CONTEXT_LENGTH)
    model = components.Model(config)

    # config = AutoConfig.from_pretrained(model_name,vocab_size=len(tokenizer),n_ctx=CONTEXT_LENGTH,bos_token_id=tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id)
    # model = AutoModelForCausalLM.from_config(config)
    # if 'gaussian' in run_name:
    #     state_dict = initialize_gpt2_weights(model, permute=False)
    # else:
    #     state_dict=model.state_dict()
    # if chkpoint is not None:
    #     save_dir = Path(
    #         f'/om2/user/ehoseini/MyData/bplm/miniberta_{data}/{model_name}/{run_name}/checkpoint_{chkpoint}')
    #     chkpnt_model=AutoModelForCausalLM.from_pretrained(save_dir)
    #     state_dict=chkpnt_model.state_dict()
    # # check if there is a previous run
    # model = AutoModelForCausalLM.from_pretrained(model_name, config=config, state_dict=state_dict)
    # #state_dict=None

    # del state_dict

    
    torch.cuda.empty_cache()
    # model.output_loss = True
    model_size = sum(t.numel() for t in model.parameters())
    print(f"{model_name} size: {model_size / 1000 ** 2:.1f}M parameters")
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
            num_training_steps=(len(train_dataloader) * train_config['num_epochs']) // gradient_accumulation_steps,
        )
    else:
        assert False

    # Pass everything to accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    # Logging variables
    batch_count = 0
    n_steps_per_epoch = math.ceil(len(train_dataloader.dataset) / train_config['batch_size'])
    # Begin training for number of epochs
    abs_step=0
    for epoch in tqdm(range(train_config['num_epochs'])):
        model.train()
        torch.cuda.empty_cache()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            batch = [torch.stack(batch[x]).transpose(1, 0) for x in ['input_ids', 'attention_mask']]
            total_loss=0
            with accelerator.accumulate(model):
                # outputs = model(batch[0], labels=batch[0], attention_mask=batch[1])
                # loss = outputs.loss
                logits, loss = model(batch[0], labels=batch[0], attention_mask=batch[1])
                total_loss+=loss
                accelerator.backward(loss)
                lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
            batch_count += len(batch)
            accelerator.log({"training_loss": total_loss}, step=abs_step)
            accelerator.log({"train/train_loss": total_loss}, step=abs_step)
            accelerator.log({"train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch},
                            step=abs_step)
            accelerator.log({"train/batch_count": batch_count}, step=abs_step)
            if abs_step % 500 == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, valid_accuracy = evaluate()
                accelerator.log({"validation/valid_loss": valid_loss}, step=abs_step)
                accelerator.log({"validation/valid_accuracy": valid_accuracy}, step=abs_step)
                save_dir = Path(
                    f'/om2/user/ehoseini/MyData/bplm/miniberta_{data}/{model_name}/{run_name}/checkpoint_{abs_step}')
                save_dir.mkdir(parents=True, exist_ok=True)
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save,
                                                state_dict=accelerator.get_state_dict(model))
                accelerator.save(
                    {
                        "epoch": epoch,"steps": abs_step,"optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                    },os.path.join(save_dir,'accelerator_states'))
                model.train()
                torch.cuda.empty_cache()
            abs_step+=1
    accelerator.end_training()
    torch.cuda.empty_cache()
