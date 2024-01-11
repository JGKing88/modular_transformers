import torch

from pathlib import Path

from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed

from modular_transformers.models.gpt2.configuration_gpt2 import GPT2Config
from modular_transformers.models import components

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import orth
import pandas as pd
import json

from sklearn.decomposition import PCA

from tqdm import tqdm

import subprocess
import os

#set seeds
torch.manual_seed(0)
np.random.seed(0)
set_seed(0)

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#set tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

import gc

