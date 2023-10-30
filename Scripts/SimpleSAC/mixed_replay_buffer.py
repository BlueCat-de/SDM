import os
import torch
import numpy as np
from SimpleSAC.utils.car_dis_comput import dist_between_cars
import ipdb
from .replay_buffer import ReplayBuffer


class MixedReplayBuffer(ReplayBuffer):
    def __init__(self, reward_scale, reward_bias, clip_action, state_dim, action_dim, realdata_path,
                 device="cuda", scale_rewards=True, scale_state=False, buffer_ratio=1,
                 residual_ratio=0.1, r_adv=None):
        super().__init__(state_dim, action_dim, device=device)

        self.scale_rewards = scale_rewards
        self.scale_state = scale_state
        self.buffer_ratio = buffer_ratio
        self.residual_ratio = residual_ratio

        # rootpath = realdata_path + "02/"
        # file_list = os.listdir(rootpath)
        file_list = []
        for f in os.listdir(realdata_path):
            for ff in os.listdir(os.path.join(realdata_path, f)):
                file_list.append(os.path.join(realdata_path, f, ff))

        file_num = []
        dataset = {'observations': [],
                   'actions': [],
                   'rewards': [],
                   'next_observations': [],
                   'terminals': []}
        key_list = ['observations', 'actions', 'rewards', 'next_observations', 'terminals']

        random_integers = np.random.randint(0, len(file_list), 20)
        newdatasets = [np.load(file_list[r], allow_pickle='TRUE').item() for r in random_integers]
        for newdataset in newdatasets:
            for key in key_list:
                dataset[key].extend(newdataset[key])

        if r_adv is not None and r_adv[0:2] == "r2":
            dataset['rewards'] = []
            length = 5
            width = 1.8
            for t in range(len(dataset['observations'])):
                ego_state = dataset['next_observations'][t][0:4]
                adv_state = dataset['next_observations'][t][4:]
                num_adv_agents = int(action_dim / 2)
                # r1
                # if r_adv == "r1":
                #     reward = 0  # 没有碰撞就没有奖励或惩罚
                #     dataset['rewards'].append(reward)
                # r2
                if r_adv[0:2] == "r2":
                    ego_col_cost_record, adv_col_cost_record, adv_road_cost_record = float('inf'), float('inf'), float(
                        'inf')
                    bv_bv_thresh = 1.5
                    bv_road_thresh = float("inf")
                    a, b, c = list(map(float, r_adv[3:].split('-')))

                    for i in range(num_adv_agents):
                        car_ego = [ego_state[0], ego_state[1],
                                   length, width, ego_state[3]]
                        car_adv = [adv_state[i * 4 + 0], adv_state[i * 4 + 1],
                                   length, width, adv_state[i * 4 + 3]]
                        dis_ego_adv = dist_between_cars(car_ego, car_adv)
                        # dis_ego_adv = math.sqrt((ego_state[0] - adv_state[i * 4 + 0]) ** 2 +
                        #                         (ego_state[1] - adv_state[i * 4 + 1]) ** 2)
                        if dis_ego_adv < ego_col_cost_record:
                            ego_col_cost_record = dis_ego_adv
                    ego_col_cost = ego_col_cost_record

                    for i in range(num_adv_agents):
                        for j in range(i + 1, num_adv_agents):
                            car_adv_j = [adv_state[j * 4 + 0], adv_state[j * 4 + 1],
                                         length, width, adv_state[j * 4 + 3]]
                            car_adv_i = [adv_state[i * 4 + 0], adv_state[i * 4 + 1],
                                         length, width, adv_state[i * 4 + 3]]
                            dis_adv_adv = dist_between_cars(car_adv_i, car_adv_j)

                            # dis_adv_adv = np.sqrt((adv_state[i * 4 + 0] - adv_state[j * 4 + 0]) ** 2 +
                            #                       (adv_state[i * 4 + 1] - adv_state[j * 4 + 1]) ** 2)
                            if dis_adv_adv < adv_col_cost_record:
                                adv_col_cost_record = dis_adv_adv
                    adv_col_cost = min(adv_col_cost_record, bv_bv_thresh)

                    road_up, road_low = 12, 0
                    car_width = 1.8
                    for i in range(num_adv_agents):
                        y = adv_state[i * 4 + 1]
                        dis_adv_road = min(road_up - (y + car_width / 2), (y - car_width / 2) - road_low)
                        if dis_adv_road < adv_road_cost_record:
                            adv_road_cost_record = dis_adv_road
                    adv_road_cost = min(adv_road_cost_record, bv_road_thresh)
                    reward = - a * ego_col_cost + b * adv_col_cost + c * adv_road_cost
                    # reward = -ego_col_cost + adv_col_cost + adv_road_cost
                    dataset['rewards'].append(reward)
                    if ego_col_cost > 15:
                        dataset['observations'] = dataset['observations'][0: t + 1]
                        dataset['terminals'] = dataset['terminals'][0: t + 1]
                        break
                # r3
                # elif r_adv == "r3":
                #     ego_col_cost_record = float('inf')
                #     for i in range(num_adv_agents):
                #         car_ego = [ego_state[0], ego_state[1],
                #                    length, width, ego_state[3]]
                #         car_adv = [adv_state[i * 4 + 0], adv_state[i * 4 + 1],
                #                    length, width, adv_state[i * 4 + 3]]
                #         dis_ego_adv = dist_between_cars(car_ego, car_adv)
                #         # dis_ego_adv = math.sqrt((ego_state[0] - adv_state[i * 4 + 0]) ** 2 +
                #         #                         (ego_state[1] - adv_state[i * 4 + 1]) ** 2)
                #         if dis_ego_adv < ego_col_cost_record:
                #             ego_col_cost_record = dis_ego_adv
                #     ego_col_cost = ego_col_cost_record
                #     reward = -ego_col_cost
                #     dataset['rewards'].append(reward)
                #     if ego_col_cost > 15:
                #         dataset['observations'] = dataset['observations'][0: t + 1]
                #         dataset['terminals'] = dataset['terminals'][0: t + 1]
                #         break

        # load expert dataset into the replay buffer
        # total_num = dataset['observations'].shape[0]
        total_num = len(dataset['observations'])
        # idx = random.sample(range(total_num), int(total_num * self.residual_ratio))
        idx = np.random.choice(range(total_num), int(total_num * self.residual_ratio), replace=False)
        s = np.vstack(np.array(dataset['observations'])).astype(np.float32)[idx, :
            ]  # An (N, dim_observation)-dimensional numpy array of observations
        a = np.vstack(np.array(dataset['actions'])).astype(np.float32)[idx,
            :]  # An (N, dim_action)-dimensional numpy array of actions
        r = np.vstack(np.array(dataset['rewards'])).astype(np.float32)[idx,
            :]  # An (N,)-dimensional numpy array of rewards
        s_ = np.vstack(np.array(dataset['next_observations'])).astype(np.float32)[idx,
             :]  # An (N, dim_observation)-dimensional numpy array of next observations
        done = np.vstack(np.array(dataset['terminals']))[idx,
               :]  # An (N,)-dimensional numpy array of terminal flags

        # whether to bias the reward
        r = r * reward_scale + reward_bias
        # whether to clip actions
        a = np.clip(a, -clip_action, clip_action)

        fixed_dataset_size = r.shape[0]
        self.fixed_dataset_size = fixed_dataset_size
        self.ptr = fixed_dataset_size
        self.size = fixed_dataset_size
        self.max_size = (self.buffer_ratio + 1) * fixed_dataset_size
        # ipdb.set_trace()
        self.state = np.vstack((s, np.zeros((self.max_size - self.fixed_dataset_size, state_dim))))
        self.action = np.vstack((a, np.zeros((self.max_size - self.fixed_dataset_size, action_dim))))
        self.next_state = np.vstack((s_, np.zeros((self.max_size - self.fixed_dataset_size, state_dim))))
        self.reward = np.vstack((r, np.zeros((self.max_size - self.fixed_dataset_size, 1))))
        self.done = np.vstack((done, np.zeros((self.max_size - self.fixed_dataset_size, 1))))
        self.device = torch.device(device)

        # # State normalization
        self.normalize_states()

    def normalize_states(self, eps=1e-3):
        # STATE: standard normalization
        self.state_mean = self.state.mean(0, keepdims=True)
        self.state_std = self.state.std(0, keepdims=True) + eps
        if self.scale_state:
            self.state = (self.state - self.state_mean) / self.state_std
            self.next_state = (self.next_state - self.state_mean) / self.state_std

    def append(self, s, a, r, s_, done):

        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.next_state[self.ptr] = s_
        self.reward[self.ptr] = r
        self.done[self.ptr] = done

        # fix the offline dataset and shuffle the simulated part
        self.ptr = (self.ptr + 1 - self.fixed_dataset_size) % (
                    self.max_size - self.fixed_dataset_size) + self.fixed_dataset_size
        self.size = min(self.size + 1, self.max_size)

    def append_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(observations, actions, rewards, next_observations, dones):
            self.append(o, a, r, no, d)

    def sample(self, batch_size, scope=None, type=None):
        if scope == None:
            ind = np.random.randint(0, self.size, size=batch_size)
        elif scope == "real":
            ind = np.random.randint(0, self.fixed_dataset_size, size=batch_size)
        elif scope == "sim":
            ind = np.random.randint(self.fixed_dataset_size, self.size, size=batch_size)
        else:
            raise RuntimeError("Misspecified range for replay buffer sampling")

        if type == None:
            return {
                'observations': torch.FloatTensor(self.state[ind]).to(self.device),
                'actions': torch.FloatTensor(self.action[ind]).to(self.device),
                'rewards': torch.FloatTensor(self.reward[ind]).to(self.device),
                'next_observations': torch.FloatTensor(self.next_state[ind]).to(self.device),
                'dones': torch.FloatTensor(self.done[ind]).to(self.device)
            }
        elif type == "sas":
            return {
                'observations': torch.FloatTensor(self.state[ind]).to(self.device),
                'actions': torch.FloatTensor(self.action[ind]).to(self.device),
                'next_observations': torch.FloatTensor(self.next_state[ind]).to(self.device)
            }
        elif type == "sa":
            return {
                'observations': torch.FloatTensor(self.state[ind]).to(self.device),
                'actions': torch.FloatTensor(self.action[ind]).to(self.device)
            }
        else:
            raise RuntimeError("Misspecified return data types for replay buffer sampling")

    def get_mean_std(self):
        return torch.FloatTensor(self.state_mean).to(self.device), torch.FloatTensor(self.state_std).to(self.device)
