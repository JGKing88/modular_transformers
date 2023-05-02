import torch

def l2_reg(tensor):
    norms = torch.linalg.vector_norm(tensor[0], axis=2)
    mean_norms = torch.mean(norms, axis=(0, 1))
    return mean_norms

