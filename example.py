import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from modular_transformers.models.gpt2.configuration_gpt2 import GPT2Config
from modular_transformers.models import components
# -----------------------------------------------------------------------------
resume = None
start = "\n" 
num_samples = 1 # number of samples to draw
max_new_tokens = 10 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if resume is not None:
    #resume points to a pkl file
    path = os.path.join('data', resume)
    if os.path.exists(path):
        print(f"Loading meta from {path}...")
        with open(path, 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError(f"Could not find {path}!")
else:
    configuration = GPT2Config()
    model = components.Model(configuration)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

model.to(device)
model.eval()

# encode the beginning of the prompt
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature)
            print(decode(y[0].tolist()))
            print('---------------')