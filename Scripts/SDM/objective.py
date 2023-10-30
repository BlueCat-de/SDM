import torch

def objective_function(p_ego, p_adv):
    ego_loss = -torch.mean(p_ego)
    adv_loss = -torch.mean(p_adv)
    return ego_loss, adv_loss