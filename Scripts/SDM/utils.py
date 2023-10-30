import torch
import numpy as np
from torch import autograd
import scipy
import ipdb

def stackup_array(arr, index):
    
    n = len(index)-1
    
    return tuple(arr[index[i]:index[i+1], :] for i in range(n))


class JacobianVectorProduct(scipy.sparse.linalg.LinearOperator):
    def __init__(self, grad, params, regularization=0, device='cuda:0'):
        if isinstance(grad, (list, tuple)):
            grad = list(grad)
            for i, g in enumerate(grad):
                grad[i] = g.view(-1)
            self.grad = torch.cat(grad)
        elif isinstance(grad, torch.Tensor):
            self.grad = grad.view(-1)

        nparams = sum(p.numel() for p in params)
        self.shape = (nparams, self.grad.size(0))
        self.dtype = np.dtype('float32')
        self.params = params
        self.regularization = regularization
        self.device = device

    def _matvec(self, v):
        v = torch.Tensor(v)
        v = v.to(self.grad.device)
        # ipdb.set_trace()
        hv = autograd.grad(self.grad, self.params, v, retain_graph=True, allow_unused=True)
        _hv = []
        for g, p in zip(hv, self.params):
            if g is None:
                g = torch.zeros_like(p)
            _hv.append(g.contiguous().view(-1))
        if self.regularization != 0:
            # ipdb.set_trace()
            hv = torch.cat(_hv) + self.regularization*v
        else:
            hv = torch.cat(_hv) 
        return hv.cpu()
    
    def _matmat(self, X):
        # ipdb.set_trace()
        return np.hstack([self._matvec(col) for col in X.T])


def build_game_gradient(fs, params):
    # ipdb.set_trace()
    grads = [autograd.grad(f, param.parameters(), create_graph=True) 
           for f,param in zip(fs, params)]
    return grads
      
def build_game_jacobian(fs, params):
    f1, f2 = fs
    x1, x2 = params
    A = JacobianVectorProduct(f1, list(x1.parameters()))
    B = JacobianVectorProduct(f2, list(x1.parameters()))
    C = JacobianVectorProduct(f1, list(x2.parameters()))
    D = JacobianVectorProduct(f2, list(x2.parameters()))
    J = JacobianVectorProduct(f1 + f2, list(x1.parameters()) + list(x2.parameters()))
            
    return A, B, C, D, J

def network_param_index(network):
    
    counts = [0]
    for p in network.parameters():
        count = 1
        for num in p.shape:
            count *= num
        counts.append(count)
    index = np.cumsum(counts)
    
    return index

def calc_gradient_norm(params):
    norm = lambda p: p.grad.data.norm(2).item()
    return sum(norm(p)**2 for p in params) ** (1. / 2)