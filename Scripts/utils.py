import random
import pprint
import time
import uuid
import tempfile
import os
import re
from copy import copy
from socket import gethostname
import pickle

import numpy as np

import absl.flags
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import config_dict
import ipdb
import wandb

import torch


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, 'automatically defined flag')
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, 'automatically defined flag')
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, 'automatically defined flag')
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, 'automatically defined flag')
        else:
            ipdb.set_trace()
            raise ValueError('Incorrect value type')
    return kwargs

class WandbLogger(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = True
        config.prefix = ''
        config.project = 'clearning_experimental_13'
        config.entity = 'ml_cat'
        config.output_dir = './experiment_output'
        config.random_delay = 0.0
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant):
        self.config = self.get_default_config(config)
        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.prefix != '':
            self.config.project = '{}--{}'.format(self.config.prefix, self.config.project)

        if self.config.output_dir == '':
            self.config.output_dir = tempfile.mkdtemp()
        else:
            self.config.output_dir = os.path.join(self.config.output_dir, self.config.experiment_id)
            os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = copy(variant)

        if 'hostname' not in self._variant:
            self._variant['hostname'] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        self.run = wandb.init(
            reinit=True,
            config=self._variant,
            project=self.config.project,
            entity=self.config.entity,
            dir=self.config.output_dir,
            id=self.config.experiment_id,
            anonymous=self.config.anonymous,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode='online' if self.config.online else 'offline',
        )

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        with open(os.path.join(self.config.output_dir, filename), 'wb') as fout:
            pickle.dump(obj, fout)

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir


# update user flags with flags_def
def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output

def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if prefix is not None:
            next_prefix = '{}.{}'.format(prefix, key)
        else:
            next_prefix = key
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=next_prefix))
        else:
            output[next_prefix] = val
    return output

def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)

class Timer(object):
    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time
    
def prefix_metrics(metrics, prefix):
    return {
        '{}/{}'.format(prefix, key): value for key, value in metrics.items()
        }

def eval(metrics, ego_policy, adv_policy, trajs):
    # ipdb.set_trace()
    metrics[f'{ego_policy}_{adv_policy}/average_return_adv'] = np.mean([np.mean(t['rewards_adv']) for t in trajs])
    metrics[f'{ego_policy}_{adv_policy}/average_return_ego'] = np.mean([np.mean(t['rewards_ego']) for t in trajs])
    metrics[f'{ego_policy}_{adv_policy}/average_traj_length'] = np.mean([len(t['rewards_adv']) for t in trajs])
    metrics[f'{ego_policy}_{adv_policy}/metrics_av_crash'] = np.mean([t["metrics_av_crash"] for t in trajs])
    metrics[f'{ego_policy}_{adv_policy}/metrics_bv_crash'] = np.mean([t["metrics_bv_crash"] for t in trajs])
    metrics[f'{ego_policy}_{adv_policy}/ACT'] = 0 if metrics[f'{ego_policy}_{adv_policy}/metrics_av_crash'] == 0 else \
        np.sum([t["collision_time"] for t in trajs]) / (metrics[f'{ego_policy}_{adv_policy}/metrics_av_crash'] * len(trajs))
    metrics['ACD'] = 0 if metrics[f'{ego_policy}_{adv_policy}/metrics_av_crash'] == 0 else \
        np.sum([t["collision_dis"] for t in trajs]) / (metrics[f'{ego_policy}_{adv_policy}/metrics_av_crash'] * len(trajs))
    metrics[f'{ego_policy}_{adv_policy}/CPS'] = (metrics[f'{ego_policy}_{adv_policy}/metrics_av_crash'] * len(trajs)) / np.sum([t["traj_time"] for t in trajs])
    metrics[f'{ego_policy}_{adv_policy}/CPM'] = (metrics[f'{ego_policy}_{adv_policy}/metrics_av_crash'] * len(trajs)) / np.sum([t["traj_dis"] for t in trajs])
    metrics[f'{ego_policy}_{adv_policy}/ego_speed'] = np.mean([np.mean(t['ego_speed']) for t in trajs])