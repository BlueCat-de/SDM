import torch
from torch import autograd
import scipy
import ipdb
from SDM.utils import JacobianVectorProduct, stackup_array, build_game_gradient
import numpy as np

def compute_leader_grad(ego:torch.nn.Module, 
                        adv:torch.nn.Module,
                        ego_loss,
                        adv_loss,
                        regularization,
                        x0,
                        device:torch.device,
                        precise = True
                        ):
    D_ego, D_adv = build_game_gradient([ego_loss, adv_loss], [ego, adv])
    # ipdb.set_trace()
    Dadv_ego = autograd.grad(ego_loss, adv.parameters(), retain_graph=True)
    DD_reg = JacobianVectorProduct(D_adv, list(adv.parameters()), regularization, device=device)
    # ipdb.set_trace()
    leader_grad, q, x0 = compute_stackelberg_grad(ego, D_ego, D_adv, Dadv_ego, DD_reg, x0 = x0, precise = precise, device = device)
    # ipdb.set_trace()
    # leader_grad_norm = torch.norm(leader_grad).item()
    # q_norm = torch.norm(q).item()
    # ego_norm = torch.norm(torch.cat([_.flatten() for _ in D_ego]).view(-1, 1)).item()

    return leader_grad

def adam_grad(leader_grad, beta1, beta2, epsilon, m, v, step, param_index):

    m.mul_(beta1).add_((1 - beta1)*leader_grad.detach())
    v.mul_(beta2).add_((1 - beta2)*leader_grad.detach()**2)
    bias1 = (1-beta1**(step))
    bias2 = (1-beta2**(step))
    leader_grad_to_stack = m/(bias1*(torch.sqrt(v)/np.sqrt(bias2) + epsilon))
    leader_grad = stackup_array(leader_grad_to_stack, param_index)

    return leader_grad

def leader_step(G, leader_grad, lr_ego, gamma_ego, step):

    exp_lr = lr_ego*gamma_ego**(step)
    for p, l in zip(G.parameters(), leader_grad):
        p.data.add_(-exp_lr*l.view(p.shape)) 
    
    return exp_lr

def compute_stackelberg_grad(ego_policy, D_ego, D_adv, Dadv_ego, DD_reg, device:torch.device, x0=None, tol=1e-6, precise = True):

    Dego_vec = torch.cat([_.flatten() for _ in D_ego]).view(-1, 1)
    Dadv_ego_vec = torch.cat([_.flatten() for _ in Dadv_ego]).view(-1, 1)
    if precise:
        # print(DD_reg._matvec(Dadv_ego_vec.cpu().detach().numpy().T[0]))
        # ipdb.set_trace()

        w, status = scipy.sparse.linalg.gmres(DD_reg, Dadv_ego_vec.cpu().detach().numpy(), x0=x0, tol=tol, restart = DD_reg.shape[0])
        assert status == 2 or status == 3
    else:
        w, status = scipy.sparse.linalg.cg(DD_reg, Dadv_ego_vec.cpu().detach().numpy(), maxiter=3)
    # ipdb.set_trace()
    q = torch.Tensor(JacobianVectorProduct(D_adv, list(ego_policy.parameters()))(w)).view(-1, 1).to(device)
    leader_grad = Dego_vec - q
    
    return leader_grad, q, w