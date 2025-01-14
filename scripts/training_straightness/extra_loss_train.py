# Standard library imports
import os
import math
import random
from pathlib import Path

# Third-party imports
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator
from accelerate.utils import LoggerType, DummyOptim, DummyScheduler
from transformers import (
    AdamW, AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,
    set_seed, AutoConfig
)

# Local imports
from modular_transformers.train.utils import Group_Texts
from modular_transformers.models.gpt2.utils import initialize_gpt2_weights
from modular_transformers.models.gpt2.configuration_gpt2 import GPT2Config
from modular_transformers.models import components

# Global constants
NUM_GPUS = 1
MAX_GPU_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
CONTEXT_LENGTH = 512

# Initialize wandb and set random seeds
wandb.login(key="a338f755915cccd861b14f29bf68601d8e1ec2c9")
torch.manual_seed(0)
np.random.seed(0)
set_seed(0)

def match_order(x, extra_loss):
    """
    Adjusts the scale of extra_loss to match the order of magnitude of x.
    
    Args:
        x: Reference value
        extra_loss: Value to be adjusted
    Returns:
        Adjusted extra_loss value
    """
    y = extra_loss.item()
    if x == 0 or y == 0:
        return 0
    order_of_x = math.floor(math.log10(abs(x)))
    order_of_y = math.floor(math.log10(abs(y)))
    scale_factor = 10 ** (order_of_x - order_of_y)
    return extra_loss * scale_factor

def evaluate(model, eval_dataloader, accelerator):
    """
    Evaluates model performance on the validation set.
    
    Returns:
        tuple: (loss, perplexity)
    """
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
        
    accelerator.print(f"validation loss: {loss.item()}, validation perplexity {perplexity.item()}")
    return loss.item(), perplexity.item()

def main(
    bottleneck_dim,
    n_layer,
    loss_hooks,
    normalize_loss,
    logit_multiplier,
    inter_multiplier,
    pretrained,
    warmup,
    gradient_clipping,
    clip_loss,
    epochs,
    extra_loss_delay,
    dropout_dict=None,
):
    """
    Main training function.
    
    Args:
        bottleneck_dim: Dimension of bottleneck layer
        n_layer: Number of transformer layers
        loss_hooks: Dictionary mapping layer numbers to loss function names
        normalize_loss: Whether to normalize the loss
        logit_multiplier: Multiplier for logit loss
        inter_multiplier: Multiplier for intermediate loss
        pretrained: Whether to use pretrained weights
        warmup: Number of warmup epochs
        gradient_clipping: Maximum gradient norm
        clip_loss: Whether to clip loss values
        epochs: Number of training epochs
        extra_loss_delay: Number of epochs to delay extra loss
        dropout_dict: Dictionary of dropout configurations
    """
    # Checkpoint configuration
    chkpoint = None
    wandb_id = "amvcstc4"
    epoch_buffer = 12

    # Training hyperparameters
    data = "100M"
    batch_size = 64
    lr = 0.0003
    train_config = {
        "lr": lr,
        "num_epochs": epochs,
        "correct_bias": True,
        "seed": 42,
        "batch_size": batch_size,
    }

    # Initialize tokenizer and load datasets
    tokenizer = AutoTokenizer.from_pretrained("gpt2", fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    path = "/om/weka/evlab/ehoseini/MyData/miniBERTa_v2/"
    grouped_pad_train = load_from_disk(
        os.path.join(path, f"miniBERTa-{data}-crunched", f"train_context_len_{CONTEXT_LENGTH}")
    )
    grouped_pad_valid = load_from_disk(
        os.path.join(path, f"miniBERTa-{data}-crunched", f"valid_context_len_{CONTEXT_LENGTH}")
    )

    # Configure accelerator for distributed training
    gradient_accumulation_steps = 1
    if train_config["batch_size"] > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = train_config["batch_size"] // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE
        accelerator = Accelerator(
            log_with="wandb", 
            gradient_accumulation_steps=gradient_accumulation_steps
        )
    else:
        accelerator = Accelerator(log_with="wandb")

    # Create data loaders
    eval_dataloader = DataLoader(grouped_pad_valid, shuffle=False, batch_size=EVAL_BATCH_SIZE)
    train_dataloader = DataLoader(grouped_pad_train, shuffle=True, batch_size=batch_size)
    del grouped_pad_train, grouped_pad_valid  # Free memory

    # Model configuration
    config = {
        "regsize": 768,
        "vocab_size": len(tokenizer),
        "n_ctx": CONTEXT_LENGTH,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bottleneck": bottleneck_dim,
        "n_layer": n_layer,
        "loss_hooks": loss_hooks,
        "normalize_loss": normalize_loss,
        "dropout_dict": dropout_dict,
        "logit_multiplier": logit_multiplier,
        "inter_multiplier": inter_multiplier,
        "n_heads": 12,
        "pretrained": pretrained,
        "warmup": warmup,
        "gradient_clipping": gradient_clipping,
        "clip_loss": clip_loss,
    }

    # Initialize model
    config = GPT2Config(config)
    model = components.LM(config)
    
    # Load pretrained weights if specified
    if pretrained:
        state_dict = torch.load("/om2/user/jackking/transformer_xray/scripts/dynamics/gpt2_weights.pt")
        model.load_state_dict(state_dict, strict=False)

    # Create model name based on configuration
    model_name = f"epochs{epochs}-{bottleneck_dim}x{n_layer}"
    if loss_hooks is not None:
        func_name = [(layer, name) for layer, name in loss_hooks.items()][0]
        model_name = f"lr{lr}-{inter_multiplier}-layer{func_name[0]}-{func_name[1]}-{model_name}"
        
        # Add additional configuration indicators to model name
        if normalize_loss:
            model_name = f"scaled-{model_name}"
        if warmup > 0:
            model_name = f"warmup{warmup}-{model_name}"
        if extra_loss_delay > 0:
            model_name = f"delay{extra_loss_delay}-{model_name}"
        if len(loss_hooks) > 1:
            model_name = f"multi-{model_name}"
        if clip_loss:
            model_name = f"clip-{model_name}"

    # Add dropout configuration to model name if specified
    if dropout_dict is not None:
        dropout_str = ""
        for layer, location in dropout_dict.items():
            for location, dropout in location.items():
                dropout_str += f"{layer}-{location}-{dropout}-"
        model_name = f"{dropout_str}{model_name}"

    if pretrained:
        model_name = f"finetuned-{model_name}"

    # Initialize wandb tracking
    train_config.update(config.get())
    project_name = "inner_reg_testing"

    # Load from checkpoint if specified
    if chkpoint is not None:
        save_dir = Path(
            f"/om2/user/jackking/MyData/mt/miniberta_{data}/{model_name}/checkpoint_{chkpoint}"
        )
        model = components.LM.from_pretrained(save_dir)
        # Resume wandb tracking
        accelerator.init_trackers(
            project_name, init_kwargs={"wandb": {"id": wandb_id, "resume": "must"}}
        )
        # Update wandb config with new epoch count
        api = wandb.Api()
        run = api.run(f"modular_transformers/curvature_testing/{wandb_id}")
        run.config["num_epochs"] = epoch_buffer + train_config["num_epochs"]
        run.update()
    else:
        # Start new wandb run
        accelerator.init_trackers(
            project_name,
            config=train_config,
            init_kwargs={"wandb": {"name": model_name}},
        )

    # Prepare model for training
    torch.cuda.empty_cache()
    model.output_loss = True
    model_size = sum(t.numel() for t in model.parameters())
    print(f"{model_name} size: {model_size / 1000 ** 2:.1f}M parameters")
    model = model.to(accelerator.device)
    print("Device:", accelerator.device)

    # Initialize optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(params=model.parameters(), lr=train_config["lr"])

    # Initialize learning rate scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=math.ceil(
                len(train_dataloader)
                * train_config["num_epochs"]
                / gradient_accumulation_steps
            ),
        )
    else:
        assert False  # DeepSpeed scheduler not supported

    # Prepare for distributed training
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    # Initialize training variables
    n_steps_per_epoch = len(train_dataloader)
    print(f"n_steps_per_epoch: {n_steps_per_epoch}")
    print(f"batch_size: {batch_size}")
    print(f"trainloader.dataset: {len(train_dataloader.dataset)}")
    print(f"trainloader: {len(train_dataloader)}")

    data_count = 0
    absolute_step = 0

    # Setup loss normalization if enabled
    if loss_hooks is not None and normalize_loss:
        normalization_batch_size = 10
        extra_losses_list = np.zeros(normalization_batch_size)
        extra_losses_list[0] = 1
        logit_losses_list = np.zeros(normalization_batch_size)
        logit_losses_list[0] = 1

    # Store original multiplier value
    orig_multiplier = inter_multiplier

    # Setup gradient clipping and loss tracking
    max_grad_norm = gradient_clipping
    prev_loss = 0

    # Adjust warmup steps
    warmup = warmup - extra_loss_delay
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

    # Main training loop
    for epoch in tqdm(range(train_config["num_epochs"])):
        model.train()
        torch.cuda.empty_cache()
        
        # Batch training loop
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # Prepare batch data
            batch = [
                torch.stack(batch[x]).transpose(1, 0)
                for x in ["input_ids", "attention_mask"]
            ]

            # Training step without extra loss hooks
            if loss_hooks is None:
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    outputs = model(batch[0], labels=batch[0], attention_mask=batch[1])
                    loss = outputs.loss
                    accelerator.backward(loss)

                    # Apply gradient clipping if enabled
                    if accelerator.sync_gradients:
                        if gradient_clipping is not None:
                            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                    # Calculate gradient norm for logging
                    gradients = []
                    for param in model.parameters():
                        gradients.append(param.grad)
                    gradient_norm = torch.norm(
                        torch.stack([torch.norm(g) for g in gradients if g is not None])
                    )

                    lr_scheduler.step()
                    optimizer.step()

                torch.cuda.empty_cache()
                
            # Training step with extra loss hooks
            else:
                with accelerator.accumulate(model):
                    optimizer.zero_grad()

                    # Forward pass with extra loss computation
                    outputs = model.forward_with_extra_loss(
                        batch[0],
                        labels=batch[0],
                        attention_mask=batch[1],
                        attn_indices=(batch[1] == 0)
                        .nonzero(as_tuple=False)
                        .t()
                        .detach()
                        .cpu(),
                    )
                    logit_loss = outputs.loss

                    # Calculate extra losses from hooks
                    extra_losses = model.output_extra_losses()
                    extra_loss = 0
                    for loss in extra_losses.values():
                        if loss is not None:
                            extra_loss += loss

                    base_logit_loss = logit_loss.item()
                    base_extra_loss = extra_loss.item()

                    # Apply loss normalization if enabled
                    if normalize_loss:
                        extra_loss = match_order(logit_loss.item(), extra_loss)

                    # Calculate warmup multiplier
                    try:
                        warmup_multiplier = 1 / (1+ math.e ** (-(8
                                    / (
                                        num_warmup_steps
                                        - math.ceil(np.log10(abs(inter_multiplier)) + 1)
                                    )
                                    * (
                                        (
                                            absolute_step - n_steps_per_epoch * extra_loss_delay
                                        ) 
                                        - num_warmup_steps / 2
                                    ))))
                    except:
                        warmup_multiplier = 1

                    # Apply extra loss delay if needed
                    if epoch < extra_loss_delay:
                        warmup_multiplier = 0

                    # Apply loss multipliers
                    logit_loss = logit_loss * logit_multiplier
                    extra_loss = extra_loss * inter_multiplier * warmup_multiplier

                    normalized_inter_loss = extra_loss.item()

                    # Apply loss clipping if enabled
                    if clip_loss:
                        if extra_loss - prev_loss > 0.001 and prev_loss != 0:
                            extra_loss = (
                                extra_loss * (prev_loss / extra_loss.item()) + 0.001
                            )
                        prev_loss = extra_loss.item()

                    # Combine losses and perform backward pass
                    loss = logit_loss + extra_loss
                    accelerator.backward(loss)

                    # Apply gradient clipping if enabled
                    if accelerator.sync_gradients:
                        if gradient_clipping is not None:
                            accelerator.clip_grad_norm_(
                                model.parameters(), max_grad_norm
                            )

                    lr_scheduler.step()
                    optimizer.step()

                    # Calculate gradient norm for logging
                    gradients = []
                    for param in model.parameters():
                        gradients.append(param.grad)
                    gradient_norm = torch.norm(
                        torch.stack([torch.norm(g) for g in gradients if g is not None])
                    )

                # Log training metrics
                accelerator.log({"train/inter_loss": base_extra_loss}, step=absolute_step)
                accelerator.log({"train/logit_loss": base_logit_loss}, step=absolute_step)
                accelerator.log({"train/warmup_multiplier": warmup_multiplier}, step=absolute_step)
                accelerator.log({"train/normalized_inter_loss": normalized_inter_loss}, step=absolute_step)

            # Common logging for both training modes
            data_count += batch[0].shape[0]
            absolute_step += 1

            accelerator.log({"train/train_loss": loss}, step=absolute_step)
            accelerator.log({"train/gradient": gradient_norm}, step=absolute_step)
            torch.cuda.empty_cache()

            # Log training progress
            accelerator.log(
                {"train/epoch": (absolute_step + 1) / n_steps_per_epoch},
                step=absolute_step,
            )
            accelerator.log({"train/data_count": data_count}, step=absolute_step)
            accelerator.log(
                {"train/learning_rate": lr_scheduler.get_last_lr()[0]},
                step=absolute_step,
            )

            # Periodic validation (every 200 steps)
            if absolute_step % 200 == 0:
                torch.cuda.empty_cache()
                if loss_hooks is not None:
                    model.remove_hooks()
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
                if loss_hooks is not None:
                    model.set_hooks()
                model.train()
                torch.cuda.empty_cache()

            # Periodic checkpoint saving (every 20000 steps)
            if absolute_step % 20000 == 0:
                save_dir = Path(
                    f"/om2/user/jackking/MyData/mt/miniberta_{data}/{model_name}/checkpoint_{absolute_step}"
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

    # Save final model checkpoint
    save_dir = Path(
        f"/om2/user/jackking/MyData/mt/miniberta_{data}/{model_name}/checkpoint_final"
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

    # Clean up
    accelerator.end_training()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set default configuration
    bottleneck_dim = 768
    normalize_loss = True
    logit_multiplier = 1
    inter_multiplier = 0.01
    gradient_clipping = 1
    clip_loss = False

    n_layer = 12
    pretrained = False
    warmup = 10
    epochs = 10
    extra_loss_delay = 0

    # Run training with curvature loss hooks on layers 5 and 6
    loss_hooks = {6: "curvature", 5: "curvature"}
    dropout_dict = None
    # dropout_dict = {layer: {"attn": dropout, "embd": dropout}}
    
    # Train with different inter_multiplier values
    for multiplier in [0.01, 0.1]:
        main(
            bottleneck_dim,
            n_layer,
            loss_hooks,
            normalize_loss,
            logit_multiplier,
            multiplier,
            pretrained,
            warmup,
            gradient_clipping,
            clip_loss,
            epochs,
            extra_loss_delay,
            dropout_dict=dropout_dict,
        )
