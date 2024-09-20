from transformers import (AutoTokenizer)

from modular_transformers.models.gpt2.configuration_gpt2 import GPT2Config
from modular_transformers.models import components
import os
import torch
from torch.nn import functional as F


CONTEXT_LENGTH = 10
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
os.environ['TOKENIZERS_PARALLELISM']='false'

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("gpt2", fast=False)
    override_vars = {'vocab_size': len(tokenizer), 'n_ctx': CONTEXT_LENGTH, 'bos_token_id': tokenizer.bos_token_id,
                     'eos_token_id': tokenizer.eos_token_id}
    config = GPT2Config(override_vars)
    model = components.LM(config)
    print(model)
    device = 'cpu'
    model = model.to(device)

    sentence = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")
    inputs = sentence[:-1]
    # labels = sentence[1:]
    # inputs = inputs.to(device)
    # outputs = model(inputs, labels=labels)
    # loss = outputs.loss
    # print(loss)

    # extra_losses = model.output_extra_losses()
    # for extra_loss in extra_losses.values():
    #     if extra_loss is not None:
    #         loss += extra_loss
    # print(loss)

    outputs = model(inputs)
    logits = outputs.logits[:, -1, :]
    # apply softmax to convert logits to (normalized) probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution
    idx_next = torch.multinomial(probs, num_samples=1)
    # append sampled index to the running sequence and continue
    print(tokenizer.decode(idx_next[0]))