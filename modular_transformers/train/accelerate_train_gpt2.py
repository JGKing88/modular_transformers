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

"""
Basic script to train a distill-gpt2 model using accelerate and grouping function.
Set config to use DeepSpeed
'accelerate config' -> enter in desired DeepSpeed configs or input path to deepspeed_config.json
'accelerate launch bplm/basic_accelerate_addedAug2022.py'
"""

MAX_GPU_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
CONTEXT_LENGTH = 1024


# Evaluate function
def evaluate(model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            batch = torch.stack(batch["input_ids"]).transpose(1, 0)
            outputs = model(batch, labels=batch)
        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.stack(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    accelerator.print(
        f"validation loss: {loss.item()}, validation perplexity {perplexity.item()}"
    )
    return loss.item(), perplexity.item()


def main():

    wandb.login(key="a338f755915cccd861b14f29bf68601d8e1ec2c9")

    # Set checkpoint if needed ------------------------------------------------
    chkpoint = None
    wandb_id = "amvcstc4"
    epoch_buffer = 12

    # Set training config --------------------------------------------

    data = "100M"
    batch_size = 64

    train_config = {
        "lr": 0.0006,
        "num_epochs": 50,
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
    gradient_accumulation_steps = 1
    if train_config["batch_size"] > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = train_config["batch_size"] // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE
        accelerator = Accelerator(
            log_with="wandb", gradient_accumulation_steps=gradient_accumulation_steps
        )
    else:
        accelerator = Accelerator(log_with="wandb")

    eval_dataloader = DataLoader(
        grouped_pad_valid, shuffle=False, batch_size=EVAL_BATCH_SIZE
    )
    train_dataloader = DataLoader(
        grouped_pad_train, shuffle=True, batch_size=batch_size
    )
    del grouped_pad_train, grouped_pad_valid

    # set model config ---------------------------------------
    bottleneck_dim = 768
    n_layer = 3

    config = {
        "regsize": 768,
        "vocab_size": len(tokenizer),
        "n_ctx": CONTEXT_LENGTH,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bottleneck": bottleneck_dim,
        "n_layer": n_layer,
    }
    config = GPT2Config(config)
    model = components.LM(config)

    run_name = "reg_loss"
    model_name = ""
    for layer in config.n_embds:
        model_name += f"{layer}-"
    model_name = model_name[:-1]

    train_config.update(config.get())

    if chkpoint is not None:
        save_dir = Path(
            f"/om2/user/jackking/MyData/mt/miniberta_{data}/{model_name}/{run_name}/checkpoint_{chkpoint}"
        )
        model = components.LM.from_pretrained(save_dir)
        accelerator.init_trackers(
            "curvature_testing",
            init_kwargs={"wandb": {"id": wandb_id, "resume": "must"}},
        )
        api = wandb.Api()
        run = api.run(f"modular_transformers/curvature_testing/{wandb_id}")
        run.config["num_epochs"] = epoch_buffer + train_config["num_epochs"]
        run.update()
    else:
        accelerator.init_trackers(
            "curvature_testing",
            config=train_config,
            init_kwargs={"wandb": {"name": model_name}},
        )

    torch.cuda.empty_cache()
    model.output_loss = True
    model_size = sum(t.numel() for t in model.parameters())
    print(f"{model_name} size: {model_size / 1000 ** 2:.1f}M parameters")
    print(model)
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
            num_training_steps=(len(train_dataloader) * train_config["num_epochs"]),
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
    n_steps_per_epoch = math.ceil(len(train_dataloader.dataset) / batch_size)
    data_count = 0
    absolute_step = 0
    # Begin training for number of epochs

    for epoch in tqdm(range(train_config["num_epochs"])):
        model.train()
        torch.cuda.empty_cache()
        for step, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            batch = [
                torch.stack(batch[x]).transpose(1, 0)
                for x in ["input_ids", "attention_mask"]
            ]

            with accelerator.accumulate(model):
                outputs = model(batch[0], labels=batch[0], attention_mask=batch[1])
                loss = outputs.loss

                # dealing with extra losses. can change by case
                # can not do this here and instead backpropagate from the middle (in components.py)

                # extra_losses = model.output_extra_losses()
                # for extra_loss in extra_losses.values():
                #     if extra_loss is not None:
                #         loss += extra_loss

                accelerator.backward(loss)
                lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

            data_count += batch[0].shape[0]

            absolute_step += 1

            accelerator.log({"train/train_loss": loss}, step=absolute_step)
            accelerator.log(
                {"train/epoch": (absolute_step + 1) / n_steps_per_epoch},
                step=absolute_step,
            )
            accelerator.log({"train/data_count": data_count}, step=absolute_step)
            accelerator.log(
                {"train/learning_rate": lr_scheduler.get_last_lr()[0]},
                step=absolute_step,
            )

            if absolute_step % 200 == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, valid_accuracy = evaluate(
                        model, eval_dataloader, accelerator
                    )
                accelerator.log(
                    {"validation/valid_loss": valid_loss}, step=absolute_step
                )
                accelerator.log(
                    {"validation/valid_accuracy": valid_accuracy}, step=absolute_step
                )
                model.train()
                torch.cuda.empty_cache()

            if absolute_step % 2000 == 0:
                save_dir = Path(
                    f"/om2/user/jackking/MyData/mt/miniberta_{data}/{model_name}/{run_name}/checkpoint_{absolute_step}"
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    save_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model),
                )
                accelerator.save(
                    {
                        "epoch": epoch,
                        "steps": step,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                    },
                    os.path.join(save_dir, "accelerator_states"),
                )

    accelerator.end_training()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
