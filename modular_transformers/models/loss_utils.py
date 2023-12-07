import torch

def l2_reg(tensor):
    #tensor[0].size(): batch size, context length, embedding sie
    norms = torch.linalg.vector_norm(tensor[0], axis=2)
    mean_norms = torch.mean(norms, axis=(0, 1))
    return mean_norms * 10

