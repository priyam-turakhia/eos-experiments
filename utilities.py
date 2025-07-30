import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
from scipy.sparse.linalg import LinearOperator, eigsh
from loss_functions import MSELoss

def compute_hvp(network: nn.Module, loss_fn: nn.Module, loader: DataLoader, vector: torch.Tensor):
    
    device = vector.device
    
    p = len(parameters_to_vector(network.parameters()))
    hvp = torch.zeros(p, device = device)
    
    for data, target in loader:

        data, target = data.to(device), target.to(device)
        preds = network(data)

        target_tensor = target

        loss = loss_fn(preds, target_tensor) / len(loader.dataset)
        grads = torch.autograd.grad(loss, inputs=list(network.parameters()), create_graph = True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads2 = [g.contiguous() for g in torch.autograd.grad(dot, inputs=list(network.parameters()), retain_graph = True)]
        hvp += parameters_to_vector(grads2)
        
    return hvp

def lanczos(matrix_vector, dim: int, neigs: int, device: torch.device):
    
    def mv(x):
        v = torch.tensor(x, dtype = torch.float, device = device)
        hv = matrix_vector(v)
        return hv.to('cpu').numpy()
    
    op = LinearOperator(dtype = np.float32, shape = (dim, dim), matvec = mv)
    evals, _ = eigsh(op, k = neigs)
    sorted_evals = np.sort(evals)[::-1].copy()

    return torch.from_numpy(sorted_evals).float()