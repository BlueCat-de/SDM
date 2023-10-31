import numpy as np
from datetime import datetime
import ipdb
import torch

class StepSampler(object):

    def __init__(self, env, max_traj_length = 1000):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0

    def sample(self, ego_policy, adv_policy, n_steps, deterministic = False, replay_buffer = None, joint_noise_std = 0.):
        # observations = []
        # actions_ego = []
        # actions_adv = []
        # rewards_ego = []
        # rewards_adv = []
        # next_observations = []
        # dones = []
        self.env.traci_start()
        self._current_observation = self.env.reset(self.env.ego_policy, self.env.adv_policy)
        for _ in range(n_steps):
            self._traj_steps += 1
            observation = self._current_observation

            # TODO: sample actions from current policy
            if ego_policy != None:
                # ipdb.set_trace()
                action_ego = ego_policy(np.expand_dims(observation, 0), deterministic = deterministic)[0, :]
                # ipdb.set_trace()
            else:
                action_ego = None

            if adv_policy != None:
                action_adv = adv_policy(np.expand_dims(observation, 0), deterministic = deterministic)[0, :]
            else:
                action_adv = None

            if joint_noise_std > 0.:
                # normal distribution
                next_observation, reward, done, _ = self.env.step(
                    action_ego + np.random.randn(action_ego.shape[0],) * joint_noise_std,
                    action_adv + np.random.randn(action_adv.shape[0],) * joint_noise_std
                                                                  )
            else:
                next_observation, reward, done, _ = self.env.step(action_ego, action_adv)

            # observations.append(observation)
            # actions_ego.append(action_ego)
            # actions_adv.append(action_adv)
            # rewards_ego.append(reward[0])
            # rewards_adv.append(reward[1])
            # dones.append(done)
            # next_observations.append(next_observation)

            # add samples derived from current policy to replay buffer
            if replay_buffer is not None:
                # ipdb.set_trace()
                replay_buffer.append(observation, (action_ego, action_adv), reward, next_observation, done)

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation = self.env.reset(self.env.ego_policy, self.env.adv_policy)
            else:
                self._current_observation = next_observation
        self.env.traci_close()
        
    @property
    def env(self):
        return self._env
    

    
class TrajSampler(object):

    def __init__(self, env, rootsavepath, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env
        self.rootsavepath = rootsavepath

    @property
    def env(self):
        return self._env

    def sample(self, ego_policy, adv_policy, n_trajs, deterministic = False, replay_buffer = None, idxes = None, av = None, bv = None):
        global info, done
        trajs = []
        self.env.traci_start()
        # ipdb.set_trace()
        for i in range(n_trajs):
            observations = []
            actions_ego = []
            actions_adv = []
            rewards_ego = []
            rewards_adv = []
            next_observations = []
            dones = []

            num_av_crash = 0
            num_bv_crash = 0
            if idxes is None:
                observation = self.env.reset(self.env.ego_policy, self.env.adv_policy, idx = None)
            else:
                observation = self.env.reset(self.env.ego_policy, self.env.adv_policy, idx = idxes[i])
            for _ in range(self.max_traj_length):
                if ego_policy != None:
                    action_ego = ego_policy(np.expand_dims(observation, 0), deterministic = deterministic)[0, :]
                else:
                    action_ego = None

                if adv_policy != None:
                    action_adv = adv_policy(np.expand_dims(observation, 0), deterministic = deterministic)[0, :]
                # TODO: what if adv_policy is not RL?
                else:
                    action_adv = None

                next_observation, reward, done, info = self.env.step(action_ego, action_adv)
                observations.append(observation)
                actions_ego.append(action_ego)
                actions_adv.append(action_adv)
                rewards_ego.append(reward[0])
                rewards_adv.append(reward[1])
                dones.append(done)
                next_observations.append(next_observation)

                if replay_buffer is not None:
                    replay_buffer.add_sample(
                        observation, np.concatenate((action_ego, action_adv)), reward, next_observation, done
                    )

                observation = next_observation
                if done:
                    break
            if av is not None and bv is not None:
                import os
                if not os.path.isdir(self.rootsavepath + f'{av}vs{bv}'):
                    os.mkdir(self.rootsavepath + f'{av}vs{bv}')
            if info[0] == "AV crashed!":
                if self.rootsavepath != "None":
                    if av is not None and bv is not None:
                        import os
                        if not os.path.isdir(self.rootsavepath + f'{av}vs{bv}/avcrash_{idxes[i]}'):
                            os.mkdir(self.rootsavepath + f'{av}vs{bv}/avcrash_{idxes[i]}')
                        # os.mkdir(self.rootsavepath + f'{av}vs{bv}/avcrash_{idxes[i]}')
                        self.env.record(filepath=self.rootsavepath + f'{av}vs{bv}/avcrash_{idxes[i]}/record-' +
                                                datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f") + '.csv')
                    else:
                        self.env.record(filepath=self.rootsavepath + f'avcrash/record-' +
                                                datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f") + '.csv')
                num_av_crash += 1

            elif info[0] == "BV crashed!":
                if self.rootsavepath != "None":
                    if av is not None and bv is not None:
                        import os
                        if not os.path.isdir(self.rootsavepath + f'{av}vs{bv}/bvcrash_{idxes[i]}'):
                            os.mkdir(self.rootsavepath + f'{av}vs{bv}/bvcrash_{idxes[i]}')
                        self.env.record(filepath=self.rootsavepath + f'{av}vs{bv}/bvcrash_{idxes[i]}/record-' +
                                                datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f") + '.csv')
                    else:
                        self.env.record(filepath=self.rootsavepath + f'bvcrash/record-' +
                                                datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f") + '.csv')
                num_bv_crash += 1
            elif info[0] == "AV arrived!" or not done:
                if self.rootsavepath != "None":
                    if av is not None and bv is not None:
                        import os
                        # os.mkdir(self.rootsavepath + f'{av}vs{bv}')
                        if not os.path.isdir(self.rootsavepath + f'{av}vs{bv}/avarrive_{idxes[i]}'):
                            os.mkdir(self.rootsavepath + f'{av}vs{bv}/avarrive_{idxes[i]}')
                        self.env.record(filepath=self.rootsavepath + f'{av}vs{bv}/avarrive_{idxes[i]}/record-' +
                                                datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f") + '.csv')
                    else:
                        self.env.record(filepath=self.rootsavepath + f'avarrive/record-' +
                                                datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f") + '.csv')
            
            traj_time = len(observations)
            traj_dis = observations[-1][0]
            collision_time = 0 if info[0] != 'AV crashed' else traj_time
            collision_dis = 0 if info[0] != "AV crashed!" else traj_dis
            speed = [row[2] for row in observations]
            trajs.append(dict(
                observations=np.array(observations, dtype=np.float32),
                ego_speed = np.array(speed, dtype=np.float32),
                # actions_ego=np.array(actions_ego, dtype=np.float32),
                # actions_adv=np.array(actions_adv, dtype=np.float32),
                rewards_ego=np.array(rewards_ego, dtype=np.float32),
                rewards_adv=np.array(rewards_adv, dtype=np.float32),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
                metrics_av_crash=num_av_crash,
                metrics_bv_crash=num_bv_crash,
                traj_time=traj_time,
                traj_dis=traj_dis,
                collision_time=collision_time,
                collision_dis=collision_dis,
            ))
        self.env.traci_close()

        return trajs, info[0]
    

class EVALTrajSampler(object):

    def __init__(self, env, rootsavepath, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env
        self.rootsavepath = rootsavepath

    @property
    def env(self):
        return self._env

    def sample(self, ego_policy_dict, adv_policy_dict, n_trajs, deterministic = False, replay_buffer = None, idxes = None):
        global info, done
        trajs = []
        
        # ipdb.set_trace()
        for i in range(n_trajs):
            info = dict()
            for av in ego_policy_dict.keys():
                for bv in adv_policy_dict.keys():
                    self.env.traci_start()
                    ego_policy = ego_policy_dict[av]
                    adv_policy = adv_policy_dict[bv]
                    observations = []
                    actions_ego = []
                    actions_adv = []
                    rewards_ego = []
                    rewards_adv = []
                    next_observations = []
                    dones = []

                    num_av_crash = 0
                    num_bv_crash = 0
                    if idxes is None:
                        observation = self.env.reset(self.env.ego_policy, self.env.adv_policy, idx = None)
                    else:
                        observation = self.env.reset(self.env.ego_policy, self.env.adv_policy, idx = idxes[i])
                    for _ in range(self.max_traj_length):
                        if ego_policy != None:
                            action_ego = ego_policy(np.expand_dims(observation, 0), deterministic = deterministic)[0, :]
                        else:
                            action_ego = None

                        if adv_policy != None:
                            action_adv = adv_policy(np.expand_dims(observation, 0), deterministic = deterministic)[0, :]
                        # TODO: what if adv_policy is not RL?
                        else:
                            action_adv = None

                        next_observation, reward, done, info[f'{av} vs {bv}'] = self.env.step(action_ego, action_adv)
                        observations.append(observation)
                        actions_ego.append(action_ego)
                        actions_adv.append(action_adv)
                        rewards_ego.append(reward[0])
                        rewards_adv.append(reward[1])
                        dones.append(done)
                        next_observations.append(next_observation)

                        if replay_buffer is not None:
                            replay_buffer.add_sample(
                                observation, np.concatenate((action_ego, action_adv)), reward, next_observation, done
                            )

                        observation = next_observation
                        
                        if done:
                            print('{} vs {} {}!'.format(av, bv, info[f'{av} vs {bv}'][0]))
                            break
            
            if info['pretrained-AV vs RL-BV'][0] == "AV crashed!" and info['RL-AV vs RL-BV'][0] == 'AV arrived!':
                if self.rootsavepath != "None":
                    self.env.record(filepath=self.rootsavepath + 'RL-AV-win/record-' +
                                            datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f") + '.csv')
                num_av_crash += 1

            # elif info[0] == "BV crashed!":
            #     if self.rootsavepath != "None":
            #         self.env.record(filepath=self.rootsavepath + 'bvcrash/record-' +
            #                                  datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f") + '.csv')
            #     num_bv_crash += 1
            # elif info[0] == "AV arrived!" or not done:
            #     if self.rootsavepath != "None":
            #         self.env.record(filepath=self.rootsavepath + 'avarrive/record-' +
            #                                  datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f") + '.csv')
            
        #     traj_time = len(observations)
        #     traj_dis = observations[-1][0]
        #     collision_time = 0 if info[0] != 'AV crashed' else traj_time
        #     collision_dis = 0 if info[0] != "AV crashed!" else traj_dis
        #     speed = [row[2] for row in observations]
        #     trajs.append(dict(
        #         observations=np.array(observations, dtype=np.float32),
        #         ego_speed = np.array(speed, dtype=np.float32),
        #         # actions_ego=np.array(actions_ego, dtype=np.float32),
        #         # actions_adv=np.array(actions_adv, dtype=np.float32),
        #         rewards_ego=np.array(rewards_ego, dtype=np.float32),
        #         rewards_adv=np.array(rewards_adv, dtype=np.float32),
        #         next_observations=np.array(next_observations, dtype=np.float32),
        #         dones=np.array(dones, dtype=np.float32),
        #         metrics_av_crash=num_av_crash,
        #         metrics_bv_crash=num_bv_crash,
        #         traj_time=traj_time,
        #         traj_dis=traj_dis,
        #         collision_time=collision_time,
        #         collision_dis=collision_dis,
        #     ))
        # 

        # return trajs