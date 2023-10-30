from ml_collections import ConfigDict
import torch
import torch.nn.functional as F
from SimpleSAC.models.model import Scalar, soft_target_update
import ipdb


class SAC(object):

    @staticmethod
    def get_default_config(updates = None):
        config = ConfigDict()
        # TODO: gamma 0.99->0.999
        config.discount = 0.99
        config.reward_scale = 1.0
        # TODO: alpha_multiplier = 0.2, use_automatic_entropy_tuning = False
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = True
        config.target_entropy = 0.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.optimizer_type = 'adam'
        config.soft_target_update_rate = 5e-3
        config.target_update_period = 1

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config
    
    def __init__(self, config, policy, qf1, qf2, target_qf1, target_qf2):
        self.config = SAC.get_default_config(config)
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        optimizer = {
            'adam' : torch.optim.Adam,
            'sgd' : torch.optim.SGD,
        }[self.config.optimizer_type]

        self.policy_optimizer = optimizer(self.policy.parameters(), self.config.policy_lr)
        self.qf_optimizer = optimizer(list(self.qf1.parameters()) + list(self.qf2.parameters()), self.config.qf_lr)

        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = optimizer(self.log_alpha.parameters(), lr=self.config.policy_lr)

        else:
            self.log_alpha = None

        self.update_target_network(1.0)
        self._total_steps = 0

    def update_target_network(self, soft_target_update_rate):
        soft_target_update(self.qf1, self.target_qf1, soft_target_update_rate)
        soft_target_update(self.qf2, self.target_qf2, soft_target_update_rate)

    def train(self, batch : dict, user = 'ego'):
        self._total_steps += 1

        observations = batch['observations']
        # actions = batch['actions'][:,0:2] if user == 'ego' else batch['actions'][:,2:]
        actions_ego = batch['actions_ego'] # has grad to ego_policy
        actions_adv = batch['actions_adv'] # has grad to adv_policy

        # ipdb.set_trace()
        rewards = batch['rewards'][:,0] if user == 'ego' else batch['rewards'][:,1]
        next_observations = batch['next_observations']
        dones = batch['dones']

        new_actions, log_pi = self.policy(observations)
        # ipdb.set_trace()
        if self.config.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (log_pi + self.config.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.config.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.config.alpha_multiplier)

        # policy loss
        # ipdb.set_trace()
        if user == 'ego':
            q_new_actions = torch.min(
                self.qf1(observations, new_actions, actions_adv),
                self.qf2(observations, new_actions, actions_adv)
            )
        else:
            q_new_actions = torch.min(
                self.qf1(observations, actions_ego, new_actions),
                self.qf2(observations, actions_ego, new_actions)
            )
        policy_loss = ((alpha * log_pi) - q_new_actions).mean() # gradient ascent

        # qf loss
        # TODO: fix shape missmatch
        # ipdb.set_trace()
        q_pred1 = self.qf1(observations, actions_ego, actions_adv)
        
        q_pred2 = self.qf2(observations, actions_ego, actions_adv)

        new_next_actions, next_log_pi = self.policy(next_observations)
        if user == 'ego':
            target_q_values = torch.min(
                self.target_qf1(next_observations, new_next_actions, actions_adv),
                self.target_qf2(next_observations, new_next_actions, actions_adv),
            )
        else:
            target_q_values = torch.min(
                self.target_qf1(next_observations, actions_ego, new_next_actions),
                self.target_qf2(next_observations, actions_ego, new_next_actions),
            )

        if self.config.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi
        # ipdb.set_trace()
        q_target = self.config.reward_scale * torch.squeeze(rewards, -1) + (1. - torch.squeeze(dones, -1)) * self.config.discount * target_q_values
        qf1_loss = F.mse_loss(q_pred1, q_target.detach())
        qf2_loss = F.mse_loss(q_pred2, q_target.detach())
        qf_loss = qf1_loss + qf2_loss

        if self.config.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.qf_optimizer.step()

        if self.total_steps % self.config.target_update_period == 0:
            self.update_target_network(
                self.config.soft_target_update_rate
            )

        return dict(
            log_pi = log_pi.mean().item(),
            policy_loss = policy_loss.item(),
            qf1_loss = qf1_loss.item(),
            qf2_loss = qf2_loss.item(),
            alpha_loss = alpha_loss.item(),
            alpha = alpha.item(),
            average_qf1 = q_pred1.mean().item(),
            average_qf2 = q_pred2.mean().item(),
            average_target_q = target_q_values.mean().item(),
            average_reward = rewards.mean().item(),
            total_steps = self.total_steps
        )
    
    def torch_to_device(self, device):
        for module in self.modules:
            module.to(device)

    @property
    def modules(self):
        modules = [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]
        if self.config.use_automatic_entropy_tuning:
            modules.append(self.log_alpha)
        return modules
    
    @property
    def total_steps(self):
        return self._total_steps