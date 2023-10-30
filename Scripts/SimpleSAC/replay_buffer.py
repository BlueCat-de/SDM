import torch
import numpy as np
import ipdb

class Buffer(object):
    def append(self, *args):
        pass
    def sample(self, *args):
        pass

class ReplayBuffer(Buffer):
    def __init__(self, state_dim, action_ego_dim, action_adv_dim, max_size=int(1e6), device='cuda'):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_ego_dim = action_ego_dim
        self.action_adv_dim = action_adv_dim
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action_ego = np.zeros((max_size, action_ego_dim))
        self.action_adv = np.zeros((max_size, action_adv_dim))
        self.next_state = np.zeros((max_size, state_dim))
        # self.next_action_ego = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 2))
        self.done = np.zeros((max_size, 1))

        self.device = torch.device(device)

    def reset(self):
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, self.state_dim))
        self.action_ego = np.zeros((self.max_size, self.action_ego_dim))
        self.action_adv = np.zeros((self.max_size, self.action_adv_dim))
        self.next_state = np.zeros((self.max_size, self.state_dim))
        # self.next_action = np.zeros((self.max_size, self.action_dim))
        self.reward = np.zeros((self.max_size, 2))
        self.done = np.zeros((self.max_size, 1))

    def append(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        # ipdb.set_trace()
        self.action_ego[self.ptr] = action[0]
        if action[1] is not None:
            self.action_adv[self.ptr] = action[1]
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
       #  ipdb.set_trace()
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size) # randomly sample batch_size number of indices

        return {
            'observations': torch.FloatTensor(self.state[ind]).to(self.device), 
            'actions_ego': torch.FloatTensor(self.action_ego[ind]).to(self.device), 
            'actions_adv': torch.FloatTensor(self.action_adv[ind]).to(self.device),
            'rewards': torch.FloatTensor(self.reward[ind]).to(self.device), 
            'next_observations': torch.FloatTensor(self.next_state[ind]).to(self.device), 
            'dones': torch.FloatTensor(self.done[ind]).to(self.device)
            }


class GradReplayBuffer(Buffer):

    def __init__(self, state_dim, action_ego_dim, action_adv_dim, max_size=int(1e6), device='cuda'):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_ego_dim = action_ego_dim
        self.action_adv_dim = action_adv_dim
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action_ego = torch.zeros((max_size, action_ego_dim))
        self.action_adv = torch.zeros((max_size, action_adv_dim))
        self.next_state = np.zeros((max_size, state_dim))
        # self.next_action_ego = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 2))
        self.done = np.zeros((max_size, 1))

        self.device = torch.device(device)

    def reset(self):
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, self.state_dim))
        self.action_ego = torch.zeros((self.max_size, self.action_ego_dim))
        self.action_adv = torch.zeros((self.max_size, self.action_adv_dim))
        self.next_state = np.zeros((self.max_size, self.state_dim))
        # self.next_action = np.zeros((self.max_size, self.action_dim))
        self.reward = np.zeros((self.max_size, 2))
        self.done = np.zeros((self.max_size, 1))

    def append(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        # ipdb.set_trace()
        self.action_ego[self.ptr] = action[0]
        if action[1] is not None:
            self.action_adv[self.ptr] = action[1]
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
       #  ipdb.set_trace()
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size) # randomly sample batch_size number of indices

        return {
            'observations': torch.FloatTensor(self.state[ind]).to(self.device), 
            'actions_ego': self.action_ego[ind].to(self.device), 
            'actions_adv': self.action_adv[ind].to(self.device),
            'rewards': torch.FloatTensor(self.reward[ind]).to(self.device), 
            'next_observations': torch.FloatTensor(self.next_state[ind]).to(self.device), 
            'dones': torch.FloatTensor(self.done[ind]).to(self.device)
            }
    
def batch_to_torch(batch, device):
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True)
        for k, v in batch.items()
    }

def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed

def parition_batch_train_test(batch, train_ratio):
    train_indices = np.random.rand(batch['observations'].shape[0]) < train_ratio
    train_batch = index_batch(batch, train_indices)
    test_batch = index_batch(batch, ~train_indices)
    return train_batch, test_batch

def subsample_batch(batch, size):
    indices = np.random.randint(batch['observations'].shape[0], size=size)
    return index_batch(batch, indices)

def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate([batch[key] for batch in batches], axis=0).astype(np.float32)
    return concatenated

def split_batch(batch, batch_size):
    batches = []
    length = batch['observations'].shape[0]
    keys = batch.keys()
    for start in range(0, length, batch_size):
        end = min(start + batch_size, length)
        batches.append({key: batch[key][start:end, ...] for key in keys})
    return batches

def split_data_by_traj(data, max_traj_length):
    dones = data['dones'].astype(bool)
    start = 0
    splits = []
    for i, done in enumerate(dones):
        if i - start + 1 >= max_traj_length or done:
            splits.append(index_batch(data, slice(start, i + 1)))
            start = i + 1

    if start < len(dones):
        splits.append(index_batch(data, slice(start, None)))

    return splits
