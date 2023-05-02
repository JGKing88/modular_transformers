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
    device = 'cpu'
    model = model.to(device)

    inputs = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model(inputs)
    # print(outputs.logits.type())
    # torch.set_printoptions(threshold=10000)
    # print(outputs.logits[0][0])
    # print(outputs.logits[0][0].size())
    # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(outputs.logits[0][0])))
    logits = outputs.logits[:, -1, :]
    # apply softmax to convert logits to (normalized) probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution
    idx_next = torch.multinomial(probs, num_samples=1)
    # append sampled index to the running sequence and continue
    print(tokenizer.decode(idx_next[0]))