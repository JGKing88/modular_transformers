import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import (
    LoggerType,
    DummyOptim,
    DummyScheduler,
    find_executable_batch_size,
)
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    set_seed,
    AutoConfig,
    get_scheduler,
)
from tqdm.auto import tqdm
import math
from pathlib import Path

MAX_GPU_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
CONTEXT_LENGTH = 1024


def train(model, optimizer, train_loader, lr_scheduler):
    model.train()
    torch.cuda.empty_cache()
    for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        batch = [
            torch.stack(batch[x]).transpose(1, 0)
            for x in ["input_ids", "attention_mask"]
        ]
        outputs = model(batch[0], labels=batch[0], attention_mask=batch[1])
        loss = outputs.loss
        accelerator.backward(loss)


def evaluate(model, eval_loader):
    model.eval()
    torch.cuda.empty_cache()
    losses = []
    for step, batch in tqdm(enumerate(eval_loader), total=len(eval_loader)):
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


@find_executable_batch_size(starting_batch_size=128)
def inner_training_loop(batch_size):
    global model, optimizer, lr_scheduler, train_dataloader  # Ensure they can be used in our context
    train_dataloader = DataLoader(
        grouped_pad_train, shuffle=True, batch_size=batch_size
    )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    train(model, optimizer, train_dataloader, lr_scheduler)


@find_executable_batch_size(starting_batch_size=128)
def inner_evaluate_loop(batch_size):
    global model, optimizer, lr_scheduler, train_dataloader, eval_dataloader  # Ensure they can be used in our context
    eval_dataloader = DataLoader(grouped_pad_valid, shuffle=True, batch_size=batch_size)
    model, optimizer, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, eval_dataloader, lr_scheduler
    )
    evaluate(model, eval_dataloader)


if __name__ == "__main__":
    run_name = "gaussian_init"
    model_name = "gpt2"
    data = "10M"
    grouped_pad_train = load_from_disk(
        os.path.join(
            "/om/user/ehoseini/MyData/miniBERTa_v2/",
            f"miniBERTa-{data}-crunched",
            f"train_context_len_{CONTEXT_LENGTH}",
        )
    )
    grouped_pad_test = load_from_disk(
        os.path.join(
            "/om/user/ehoseini/MyData/miniBERTa_v2/",
            f"miniBERTa-{data}-crunched",
            f"test_context_len_{CONTEXT_LENGTH}",
        )
    )
    grouped_pad_valid = load_from_disk(
        os.path.join(
            "/om/user/ehoseini/MyData/miniBERTa_v2/",
            f"miniBERTa-{data}-crunched",
            f"valid_context_len_{CONTEXT_LENGTH}",
        )
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2", fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(
        model_name,
        vocab_size=len(tokenizer),
        n_ctx=CONTEXT_LENGTH,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = AutoModelForCausalLM.from_config(config)
    accelerator = Accelerator()
    train_config = {
        "lr": 0.0006,
        "num_epochs": 50,
        "correct_bias": True,
        "seed": 42,
        "batch_size": 64,
    }

    tokenizer = AutoTokenizer.from_pretrained("gpt2", fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(params=model.parameters(), lr=train_config["lr"])
    # Unexpected argument error
    # correct_bias=train_config['correct_bias'])

    # Creates Dummy Scheduler if `scheduler` was spcified in the config file else creates `args.lr_scheduler_type` Scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=100, num_training_steps=10000
    )
    inner_training_loop()
