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

# from modular_transformers.train.utils import Group_Texts
from modular_transformers.train.utils import Group_Texts
from pathlib import Path
import sys

from modular_transformers.models.gpt2.utils import initialize_gpt2_weights
import pickle

"""
Basic script to train a distill-gpt2 model using accelerate and grouping function.
Set config to use DeepSpeed
'accelerate config' -> enter in desired DeepSpeed configs or input path to deepspeed_config.json
'accelerate launch bplm/basic_accelerate_addedAug2022.py'
"""

MAX_GPU_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
CONTEXT_LENGTH = 1024


# Evaluate function
def evaluate():
    model.eval()
    losses = []
    # Sets up progress bar, for loop for finding loss and perplexity
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            batch_pt = [torch.stack(batch[x]) for x in batch.keys()]
            outputs = model(
                batch_pt[0].transpose(1, 0), labels=batch_pt[2].transpose(1, 0)
            )
            # outputs = model(batch["input_ids"], labels=batch["input_ids"])
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


# Tokenizer function
# No extra implementations as padding is done through Group_Texts
def tokenize_function(examples):
    outputs = tokenizer(examples["text"])
    return outputs


torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
if __name__ == "__main__":
    run_name = "gaussian_init"
    model_name = "gpt2"
    data = "10M"
    train_config = {
        "lr": 0.0006,
        "num_epochs": 50,
        "correct_bias": True,
        "seed": 42,
        "batch_size": 64,
    }

    tokenizer = AutoTokenizer.from_pretrained("gpt2", fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare data and tokenize
    # Define the path to the miniberta_v2.py, also in the miniberta_v2_py change dir_path to where the main miniberta folder is
    if os.path.exists(
        os.path.join(
            "/om/user/ehoseini/MyData/miniBERTa_v2/",
            f"miniBERTa-{data}-crunched",
            f"train_context_len_{CONTEXT_LENGTH}",
        )
    ):
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
    else:
        """
        Instantiate a class of Group_Texts for each dataset
            4 modes possible are:
                1. Default: Divides the tokens into sequence length, drops any remainder
                2. Padding only: Divides tokens into sequence length, adds padding tokens to remainder tokens
                3. Stride only: Moves forward by the input stride, creates overlapping tokens in consecutive rows.
                                Drops remainder tokens that do not fit into the sequence length at the last stride
                                Use test_bool = True (false by default) to mask overlapping tokens
                4. Padding and Stride: moves forward by input stride, adds padding tokens to remainder tokens
                                Use test_bool = True (false by default) to mask overlapping tokens

            Parameters include: sequence length[int], stride[int], padding[bool], padding token[int], test_bool[bool], batch_size[int]

            Use batch size of 1 to send row by row of tokenized_dataset to create grouped datasets. Using batch size of 1
            and enabling padding will capture all of the original text with padding. Higher number of batches take less time.
            Batch size of 1000 is default.
            Padding token is by default the tokenizer's eos token.
            Test_bool is only needed for any modes with striding to mask any overlapping tokens (default is unmasked)
        """
        mini_dataset = load_dataset(
            "/om/user/ehoseini/MyData/miniBERTa_v2/", f"miniBERTa-{data}"
        )
        tokenized_datasets = mini_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        grouped_train = Group_Texts(
            tokenized_datasets["train"],
            tokenizer,
            seq_len=CONTEXT_LENGTH,
            padding=True,
            batch_size=500,
        )
        grouped_test = Group_Texts(
            tokenized_datasets["test"],
            tokenizer,
            seq_len=CONTEXT_LENGTH,
            padding=True,
            batch_size=500,
        )
        grouped_valid = Group_Texts(
            tokenized_datasets["validation"],
            tokenizer,
            seq_len=CONTEXT_LENGTH,
            padding=True,
            batch_size=500,
        )

        # Call the grouping function with Group_Texts
        grouped_pad_train = grouped_train.group_texts()
        grouped_pad_train.save_to_disk(
            os.path.join(
                "/om/user/ehoseini/MyData/miniBERTa_v2/",
                f"miniBERTa-{data}-crunched",
                f"train_context_len_{CONTEXT_LENGTH}",
            )
        )
        print(
            f"Total number of tokens in training dataset: {grouped_pad_train.shape[0]*CONTEXT_LENGTH}"
        )
        grouped_pad_test = grouped_test.group_texts()
        grouped_pad_test.save_to_disk(
            os.path.join(
                "/om/user/ehoseini/MyData/miniBERTa_v2/",
                f"miniBERTa-{data}-crunched",
                f"test_context_len_{CONTEXT_LENGTH}",
            )
        )
        print(
            f"Total number of tokens in test dataset: {grouped_pad_test.shape[0]*CONTEXT_LENGTH}"
        )
        grouped_pad_valid = grouped_valid.group_texts()
        grouped_pad_valid.save_to_disk(
            os.path.join(
                "/om/user/ehoseini/MyData/miniBERTa_v2/",
                f"miniBERTa-{data}-crunched",
                f"valid_context_len_{CONTEXT_LENGTH}",
            )
        )

        print(
            f"Total number of tokens in validation dataset: {grouped_pad_valid.shape[0]*CONTEXT_LENGTH}"
        )

    train_dataloader = DataLoader(
        grouped_pad_train, shuffle=True, batch_size=train_config["batch_size"]
    )
    eval_dataloader = DataLoader(
        grouped_pad_valid, shuffle=False, batch_size=EVAL_BATCH_SIZE
    )
    test_dataloader = DataLoader(
        grouped_pad_test, shuffle=True, batch_size=EVAL_BATCH_SIZE
    )

    del grouped_pad_train, grouped_pad_valid, grouped_pad_test
    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if train_config["batch_size"] > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = train_config["batch_size"] // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    accelerator = Accelerator(
        log_with="wandb", gradient_accumulation_steps=gradient_accumulation_steps
    )

    # Logging initialization
    # Change name test to log to different project
    accelerator.init_trackers("Logging_test", config=train_config)
    # Define model
    # seed = 1
    # set_seed(seed)
    config = AutoConfig.from_pretrained(
        model_name,
        vocab_size=len(tokenizer),
        n_ctx=CONTEXT_LENGTH,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = AutoModelForCausalLM.from_config(config)
    state_dict = initialize_gpt2_weights(model, permute=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=config, state_dict=state_dict
    )
    del state_dict
    torch.cuda.empty_cache()
    model.output_loss = True
    model_size = sum(t.numel() for t in model.parameters())
    print(f"{model_name} size: {model_size / 1000 ** 2:.1f}M parameters")
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
    # Unexpected argument error
    # correct_bias=train_config['correct_bias'])

    # Creates Dummy Scheduler if `scheduler` was spcified in the config file else creates `args.lr_scheduler_type` Scheduler
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
        # lr_scheduler = DummyScheduler(
        #    optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
        # )

    # Pass everything to accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    # Logging variables
    batch_count = 0
    n_steps_per_epoch = math.ceil(
        len(train_dataloader.dataset) / train_config["batch_size"]
    )

    # Begin training for number of epochs
    abs_step = 0
    for epoch in tqdm(range(train_config["num_epochs"])):
        model.train()
        for step, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            # batch.to(accelerator.device)
            True
            batch = [
                torch.stack(batch[x]).transpose(1, 0)
                for x in ["input_ids", "attention_mask"]
            ]
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                outputs = model(batch[0], labels=batch[0], attention_mask=batch[1])
                accelerator.backward(loss)
                optimizer.step()
                # scheduler.step()

            torch.cuda.empty_cache()

            loss = outputs.loss
            loss = loss / gradient_accumulation_steps

            # Log loss, epoch, batch_count to wandb
            if step + 1 < n_steps_per_epoch:
                batch_count += len(batch)
                accelerator.log({"train/train_loss": loss}, step=abs_step)
                accelerator.log(
                    {
                        "train/epoch": (step + 1 + (n_steps_per_epoch * epoch))
                        / n_steps_per_epoch
                    },
                    step=abs_step,
                )
                accelerator.log({"train/batch_count": batch_count}, step=abs_step)

            accelerator.backward(loss)

            # Step forward
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if abs_step % 250 == 0:
                accelerator.print(f"batch {abs_step}: ", loss.item())
                model.eval()
                with torch.no_grad():
                    valid_loss, valid_accuracy = evaluate()
                # Log validation loss and accuracy to wandb
                accelerator.log({"validation/valid_loss": valid_loss}, step=abs_step)
                accelerator.log(
                    {"validation/valid_accuracy": valid_accuracy}, step=abs_step
                )

                save_dir = Path(
                    f"/om2/user/ehoseini/MyData/bplm/miniberta_{data}/distilgpt2/{run_name}/checkpoint_{abs_step}"
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    save_dir,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model),
                )
                model.train()
            abs_step += 1
    # Needed for ending logging
    accelerator.end_training()

    torch.cuda.empty_cache()
