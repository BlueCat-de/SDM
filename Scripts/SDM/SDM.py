import torch
from ml_collections import ConfigDict
import ipdb
import torch.nn.functional as F
from SDM.leaderUpdate import compute_leader_grad, leader_step, adam_grad, compute_stackelberg_grad
from SDM.objective import objective_function
from SDM.utils import network_param_index, calc_gradient_norm
from SimpleSAC.models.model import soft_target_update, Scalar

class SDM(object):
    '''implementation of Stackelberg Policy Gradient'''

    @staticmethod
    def get_default_config(updates = None):
        config = ConfigDict()
        config.discount = 0.99
        config.reward_scale = 0.9
        config.backup_entropy = True
        config.reg_scale = 0.1
        config.policy_lr = 3e-4
        config.optimizer_type = 'adam'
        config.use_automatic_entropy_tuning = True
        config.alpha_multiplier = 1.0
        config.soft_target_update_rate = 5e-3
        config.target_update_period = 1
        config.target_entropy = 0.0

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, ego_policy, qf1_ego, target_qf1_ego, qf2_ego, target_qf2_ego, adv_policy, qf1_adv, target_qf1_adv, qf2_adv, target_qf2_adv, regularization = 10000., device = 'cuda:0', reg_scale = 0, use_automatic_entropy_tuning = True, backup_entropy = True):
        self.config = SDM.get_default_config(config)
        self.ego_policy = ego_policy
        self.adv_policy = adv_policy
        self.qf1_ego = qf1_ego
        self.target_qf1_ego = target_qf1_ego
        self.qf2_ego = qf2_ego
        self.target_qf2_ego = target_qf2_ego
        self.qf1_adv = qf1_adv
        self.target_qf1_adv = target_qf1_adv
        self.qf2_adv = qf2_adv
        self.target_qf2_adv = target_qf2_adv
        self.lr_ego = 2e-4
        self.lr_adv = 2e-4
        self.regularization = regularization
        self.leader_param_index = network_param_index(ego_policy)
        self.config.reg_scale = reg_scale
        self.config.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.config.backup_entropy = backup_entropy

        n_ego = sum(p.numel() for p in ego_policy.parameters())
        n_adv = sum(p.numel() for p in adv_policy.parameters())
        # ipdb.set_trace()
        self.m = torch.zeros(n_ego, 1).to(device)
        self.v = torch.zeros(n_ego, 1).to(device)
        self.beta1= 0.5
        self.beta2 = 0.999
        self.epsilon = 10**(-8)
        self.gamma_ego = 0.99999
        self.gamma_adv = 0.9999999
        self._total_steps = 0
        self.device = device

        self.opt_adv = torch.optim.Adam(self.adv_policy.parameters(), lr=self.lr_adv, betas=(self.beta1, self.beta2))
        self.sch_adv = torch.optim.lr_scheduler.ExponentialLR(self.opt_adv, gamma=self.gamma_adv)
        self.opt_qf_ego = torch.optim.Adam(list(self.qf1_ego.parameters()) + list(self.qf2_ego.parameters()), lr=self.lr_ego)
        self.opt_qf_adv = torch.optim.Adam(list(self.qf1_adv.parameters()) + list(self.qf2_adv.parameters()), lr=self.lr_adv)

        if self.config.use_automatic_entropy_tuning:
            self.log_alpha_ego = Scalar(0.0)
            self.log_alpha_adv = Scalar(0.0)
            self.alpha_optimizer_ego = torch.optim.Adam(self.log_alpha_ego.parameters(), lr=self.config.policy_lr)
            self.alpha_optimizer_adv = torch.optim.Adam(self.log_alpha_adv.parameters(), lr=self.config.policy_lr)
        else:
            self.log_alpha = None

        self.update_target_network(1.0)

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qf1_ego, self.target_qf1_ego, soft_target_update_rate)
        soft_target_update(self.qf2_ego, self.target_qf2_ego, soft_target_update_rate)
        soft_target_update(self.qf1_adv, self.target_qf1_adv, soft_target_update_rate)
        soft_target_update(self.qf2_adv, self.target_qf2_adv, soft_target_update_rate)


    def train(self, batch : dict, freeze_ego = False, freeze_adv = False):
        self._total_steps += 1
        observations = batch['observations']
        # ipdb.set_trace()
        actions_ego = batch['actions_ego'] # has grad to ego_policy
        
        actions_adv = batch['actions_adv'] # has grad to adv_policy
        # ipdb.set_trace()
        rewards = batch['rewards']
        rewards_ego = rewards[:, 0] # to maximize
        rewards_adv = rewards[:, 1] # to maximize
        # ipdb.set_trace()
        next_observations = batch['next_observations'] # observations after taking actions from current policy, a function of (ego_policy, adv_policy)
        dones = batch['dones']
        
        


        # ipdb.set_trace()
        
        new_actions_ego, log_pi_ego = self.ego_policy(observations)
        new_actions_adv, log_pi_adv = self.adv_policy(observations)

        if self.config.use_automatic_entropy_tuning:
            alpha_loss_ego = -(self.log_alpha_ego() * (log_pi_ego + self.config.target_entropy).detach()).mean()
            alpha_loss_adv = -(self.log_alpha_adv() * (log_pi_adv + self.config.target_entropy).detach()).mean()
            alpha_ego = self.log_alpha_ego().exp() * self.config.alpha_multiplier
            alpha_adv = self.log_alpha_adv().exp() * self.config.alpha_multiplier
        else:
            alpha_loss_ego = observations.new_tensor(0.0)
            alpha_ego = observations.new_tensor(self.config.alpha_multiplier)
            alpha_loss_adv = observations.new_tensor(0.0)
            alpha_adv = observations.new_tensor(self.config.alpha_multiplier)

        # ipdb.set_trace()
        q_new_actions_ego = torch.min(
            self.qf1_ego(observations, new_actions_ego, new_actions_adv),
            self.qf2_ego(observations, new_actions_ego, new_actions_adv)
        )
        q_new_actions_adv = torch.min(
            self.qf1_adv(observations, new_actions_ego.detach(), new_actions_adv),
            self.qf2_adv(observations, new_actions_ego.detach(), new_actions_adv)
        )
        
        ego_policy_loss = ((alpha_ego * log_pi_ego) - q_new_actions_ego).mean()
        adv_policy_loss = ((alpha_adv * log_pi_adv) - q_new_actions_adv - self.config.reg_scale * q_new_actions_ego).mean()

        # ipdb.set_trace()
        q1_pred_ego = self.qf1_ego(observations, actions_ego.detach(), actions_adv.detach())
        q2_pred_ego = self.qf2_ego(observations, actions_ego.detach(), actions_adv.detach())
        q1_pred_adv = self.qf1_adv(observations, actions_ego.detach(), actions_adv.detach())
        q2_pred_adv = self.qf2_adv(observations, actions_ego.detach(), actions_adv.detach())

        new_next_actions_ego, next_log_pi_ego = self.ego_policy(next_observations)
        new_next_actions_adv, next_log_pi_adv = self.adv_policy(next_observations)

        target_q_values_ego = torch.min(
            self.target_qf1_ego(next_observations, new_next_actions_ego, new_next_actions_adv),
            self.target_qf2_ego(next_observations, new_next_actions_ego, new_next_actions_adv)
        )
        target_q_values_adv = torch.min(
            self.target_qf1_adv(next_observations, new_next_actions_ego, new_next_actions_adv),
            self.target_qf2_adv(next_observations, new_next_actions_ego, new_next_actions_adv)
        )


        if self.config.backup_entropy:
            target_q_values_adv = target_q_values_adv - alpha_adv * next_log_pi_adv
            target_q_values_ego = target_q_values_ego - alpha_ego * next_log_pi_ego

        q_target_ego = self.config.reward_scale * torch.squeeze(rewards_ego, -1) + (1. - torch.squeeze(dones, -1)) * self.config.discount * target_q_values_ego
        qf1_ego_loss = F.mse_loss(q1_pred_ego, q_target_ego.detach()) # to minimize
        qf2_ego_loss = F.mse_loss(q2_pred_ego, q_target_ego.detach()) # to minimize
        qf_ego_loss = qf1_ego_loss + qf2_ego_loss

        # TODO: add regularization
        q_target_adv = self.config.reward_scale * torch.squeeze(rewards_adv, -1) + (1. - torch.squeeze(dones, -1)) * self.config.discount * target_q_values_adv

        qf1_adv_loss = F.mse_loss(q1_pred_adv, q_target_adv.detach()) # to minimize
        qf2_adv_loss = F.mse_loss(q2_pred_adv, q_target_adv.detach()) # to minimize
        qf_adv_loss = qf1_adv_loss + qf2_adv_loss

        if self.config.use_automatic_entropy_tuning:
            self.alpha_optimizer_adv.zero_grad()
            alpha_loss_adv.backward(retain_graph = True)
            self.alpha_optimizer_adv.step()

            self.alpha_optimizer_ego.zero_grad()
            alpha_loss_ego.backward(retain_graph = True)
            self.alpha_optimizer_ego.step()
            
        
        # leader update
        ego_grad = compute_leader_grad(self.ego_policy, 
                                    self.adv_policy, 
                                    ego_policy_loss, 
                                    adv_policy_loss, 
                                    self.regularization, 
                                    x0 = None, 
                                    device = self.device,
                                    precise = False)
        

        leader_grad = adam_grad(ego_grad, self.beta1, self.beta2, self.epsilon, self.m, self.v, self._total_steps, self.leader_param_index)
        
        # optimize adv
        
        if not freeze_adv:
            self.opt_adv.zero_grad()
            # ipdb.set_trace()
            adv_policy_loss.backward(retain_graph = True)
            self.opt_adv.step()



        # optimize ego
        if not freeze_ego:
            self.ego_policy.zero_grad()
            lr_ego_state = leader_step(self.ego_policy, leader_grad, self.lr_ego, self.gamma_ego, self._total_steps)

        if self.total_steps % self.config.target_update_period == 0:
            self.update_target_network(
                self.config.soft_target_update_rate
            )
        
        if not freeze_ego:
            self.opt_qf_ego.zero_grad()
            qf_ego_loss.backward(retain_graph = True)
            self.opt_qf_ego.step()

        if not freeze_adv:
            
            self.opt_qf_adv.zero_grad()
            qf_adv_loss.backward(retain_graph = True)
            self.opt_qf_adv.step()


        

        return dict(
            log_pi_ego = log_pi_ego.mean().item(),
            log_pi_adv = log_pi_adv.mean().item(),
            qf_adv_loss = qf_adv_loss.mean().item(),
            qf_ego_loss = qf_ego_loss.mean().item(),
            q1_pred_ego = q1_pred_ego.mean().item(),
            q1_pred_adv = q1_pred_adv.mean().item(),
            q2_pred_ego = q2_pred_ego.mean().item(),
            q2_pred_adv = q2_pred_adv.mean().item(),
            q_target_ego = q_target_ego.mean().item(),
            q_target_adv = q_target_adv.mean().item(),
            ego_policy_loss = ego_policy_loss.mean().item(),
            adv_policy_loss = adv_policy_loss.mean().item(),
            average_reward_ego = rewards_ego.mean().item(),
            average_reward_adv = rewards_adv.mean().item(),
            total_steps = self.total_steps
        )
    

    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        modules = [self.ego_policy, self.adv_policy, self.qf1_ego, 
                   self.qf2_ego, self.qf1_adv, self.qf2_adv, self.target_qf1_ego, 
                   self.target_qf2_ego, self.target_qf1_adv, self.target_qf2_adv]
        return modules
    
    @property
    def total_steps(self):
        return self._total_steps
    

        