import torch

base_threshold = 0.05

def l2_reg(tensor):
    #tensor.size(): batch size, context length, embedding size
    norms = torch.linalg.vector_norm(tensor, axis=2)
    mean_norms = torch.mean(norms, axis=(0, 1))
    return mean_norms

def l1_reg(tensor):
    #tensor.size(): batch size, context length, embedding size
    norms = torch.linalg.vector_norm(tensor, ord=1, axis=2)
    mean_norms = torch.mean(norms, axis=(0, 1))
    return mean_norms

def l0_reg(tensor):
    #tensor.size(): batch size, context length, embedding size
    norms = torch.linalg.vector_norm(tensor, ord=0, axis=2)
    mean_norms = torch.mean(norms, axis=(0, 1))
    return mean_norms

# def compute_layer_curvature(activations):

#     sent_act = torch.diff(activations)
    
#     # Initialize curve without in-place modification
#     curve_values = []
#     for idx in range(sent_act.shape[1] - 1):
#         dot_product = sent_act[:, idx, :] @ sent_act[:, idx + 1, :].T
#         # Normalize dot product to be within the range [-1, 1] to avoid domain errors with arccos
#         dot_product = dot_product / (torch.norm(sent_act[:, idx, :]) * torch.norm(sent_act[:, idx + 1, :]))
#         dot_product = torch.clamp(dot_product, -1, 1)
#         curve_values.append(dot_product)

#     #put curve into backprop graph
#     curve = torch.stack(curve_values)
#     curvature = torch.arccos(curve).mean()
#     return curvature

def compute_layer_curvature(activations):

    sent_act = [torch.diff(x, axis=0) for x in activations]
    # sent_act = [normalized(x) for x in sent_act]
    curvature = []
    for idy, vec in (enumerate(sent_act)):
        curve = [torch.dot(vec[idx, :], vec[idx + 1, :]) for idx in range(vec.shape[0] - 1)]
        curvature.append(torch.arccos(curve))
    curvature = [torch.mean(x) for x in curvature]
    return torch.mean(curvature)


def l2_curvature(tensor, attn_mask=None):
    #tensor.size(): batch size, context length, embedding size

    #check if tensor is a tuple
    if isinstance(tensor, tuple):
        tensor = tensor[0]

    if attn_mask is not None:
        mask_matrix = torch.ones_like(tensor)
        mask_matrix[attn_mask[0], attn_mask[1]] = 0
        tensor = tensor * mask_matrix
    
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)

    # Create shifted tensors
    token_0s = tensor[:, :-2, :] 
    token_1s = tensor[:, 1:-1, :]
    token_2s = tensor[:, 2:, :] 

    # Compute curvature
    curvature = l2_reg(token_2s - 2 * token_1s + token_0s)
    # curvature = compute_layer_curvature(tensor)

    return curvature


def l0_curvature(tensor, attn_mask=None):
    #tensor.size(): batch size, context length, embedding size

    #check if tensor is a tuple
    if isinstance(tensor, tuple):
        tensor = tensor[0]

    if attn_mask is not None:
        mask_matrix = torch.ones_like(tensor)
        mask_matrix[attn_mask[0], attn_mask[1]] = 0
        tensor = tensor * mask_matrix

    #create shifted tensors
    token_0s = tensor[:, :-2] 
    token_1s = tensor[:, 1:-1]  
    token_2s = tensor[:, 2:] 

    # curvature_sparsity = l0_reg(token_2s - 2 * token_1s + token_0s)
    curve = token_2s - 2 * token_1s + token_0s
    curve = torch.abs(curve)
    sigmoid = torch.sigmoid(50 * (curve - base_threshold))
    sigmoid_sum = torch.sum(sigmoid, axis=2)
    sigmoid_mean = sigmoid_sum.mean()

    return sigmoid_mean

def sparsity(tensor, attn_mask=None):
    #check if tensor is a tuple
    if isinstance(tensor, tuple):
        tensor = tensor[0]

    if attn_mask is not None:
        mask_matrix = torch.ones_like(tensor)
        mask_matrix[attn_mask[0], attn_mask[1]] = 0
        tensor = tensor * mask_matrix
    
    threshold = base_threshold * torch.mean(torch.abs(tensor))
    sigmoid_result = torch.sigmoid(50 * (torch.abs(tensor) - threshold))
    sparsity = torch.sum(sigmoid_result, axis=2)
    sparsity_mean = sparsity.mean()

    return sparsity_mean



def l0_curvature_max(tensor):
    #tensor.size(): batch size, context length, embedding size

    # Create shifted tensors
    token_0s = tensor[:, :-2] 
    token_1s = tensor[:, 1:-1]  
    token_2s = tensor[:, 2:] 

    # Compute curvature
    tensor = token_2s - 2 * token_1s + token_0s

    norms = torch.linalg.vector_norm(tensor, ord=0, axis=2)
    max_norms = torch.max(norms)
    return max_norms

def l1_curvature(tensor, attn_mask=None):
    #tensor.size(): batch size, context length, embedding size

    if attn_mask is not None:
        mask_matrix = torch.ones_like(tensor)
        mask_matrix[attn_mask[0], attn_mask[1]] = 0
        tensor = tensor * mask_matrix

    # Create shifted tensors
    token_0s = tensor[:, :-2] 
    token_1s = tensor[:, 1:-1]  
    token_2s = tensor[:, 2:] 

    # Compute curvature
    curvature_sparsity = l1_reg(token_2s - 2 * token_1s + token_0s)

    return curvature_sparsity

def curvature(tensor, attn_mask=None):

    if isinstance(tensor, tuple):
        tensor = tensor[0]

    if attn_mask is not None:
        mask_matrix = torch.ones_like(tensor)
        mask_matrix[attn_mask[0], attn_mask[1]] = 0
        tensor = tensor * mask_matrix

    token_0s = tensor[:, :-2]
    token_1s = tensor[:, 1:-1]  
    token_2s = tensor[:, 2:]

    cosine_sim = 1 - torch.nn.functional.cosine_similarity(token_2s - token_1s, token_1s - token_0s, dim=2)
    cosine_sim_mean = torch.mean(cosine_sim, axis=(0, 1))

    return cosine_sim_mean