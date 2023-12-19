import argparse
import numpy as np
import absl.app
import absl.flags
import torch
import wandb
import os

from datetime import datetime

from copy import deepcopy
from tqdm import trange
from viskit.logging import logger, setup_logger
from utils import define_flags_with_default, WandbLogger, get_user_flags, set_random_seed, Timer, prefix_metrics, Eval


from SDM.SDM import SDM
from SimpleSAC.sac import SAC
from SimpleSAC.envs import Env
from SimpleSAC.sampler import StepSampler, TrajSampler
from SimpleSAC.replay_buffer import ReplayBuffer, GradReplayBuffer
from SimpleSAC.models.model import TanhGaussianPolicy, SamplerPolicy, FullyConnectedQFunction


parser = argparse.ArgumentParser()
parser.add_argument('--used_wandb', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model_name', type=str, default='sumo_vs_re2h2o')
parser.add_argument('--num_agents', type=int, default=5)
parser.add_argument('--ego_policy', type=str, default='sumo', choices=['sumo'])
parser.add_argument('--adv_policy', type=str, default='re2h2o', choices=['re2h2o', 'sumo'])
parser.add_argument('--r_ego', type = str, default = 'stackelberg', choices = ['r1', 'stackelberg'])
parser.add_argument('--r_adv', type = str, default = 'stackelberg3')
parser.add_argument('--n_epochs', type = int, default = 100)
parser.add_argument('--n_loops', type = int, default = 20)

args = parser.parse_args()
args.used_wandb = True if args.used_wandb == 'True' else False

realdata_paths = os.listdir('../datasets/dataset/')
def extract_last_digit(path):
    # This function extracts the last digit from a string and returns it as an integer.
    return int(path[-1])

# Sort the realdata_paths list based on the last digit in ascending order.
sorted_realdata_paths = sorted(realdata_paths, key=extract_last_digit)
realdata_path = os.path.join('../datasets/dataset', sorted_realdata_paths[args.num_agents - 1])


FLAGS_DEF = define_flags_with_default(
    model_name=args.model_name,
    used_wandb=args.used_wandb,
    device=args.device,
    seed=args.seed,
    num_agents=args.num_agents,
    ego_policy=args.ego_policy,
    adv_policy=args.adv_policy,
    r_ego = args.r_ego,
    r_adv = args.r_adv,
    realdata_path=realdata_path,
    logging = WandbLogger.get_default_config(),
    n_epochs = args.n_epochs,
    n_loops = args.n_loops,
    max_traj_length=100,
    eval_n_trajs=20,
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
)


def main(argv):
    FLAGS = absl.flags.FLAGS
    
    if FLAGS.used_wandb:
        variant = get_user_flags(FLAGS, FLAGS_DEF)
        wandb_logger = WandbLogger(config=FLAGS.logging, variant=variant, seed = FLAGS.seed)
        wandb.run.name = f"{FLAGS.model_name}" \
                        f"bv={FLAGS.num_agents}-{FLAGS.adv_policy}_" \
                        f"seed={FLAGS.seed}_time={FLAGS.current_time}"
        setup_logger(
            variant=variant,
            exp_id=wandb_logger.experiment_id,
            seed=FLAGS.seed,
            base_log_dir=FLAGS.logging.output_dir,
            include_exp_prefix_sub_dir=False
        )
    
    set_random_seed(FLAGS.seed)
    env = Env(realdata_path=FLAGS.realdata_path, num_agents=FLAGS.num_agents, sim_horizon=FLAGS.max_traj_length,
                ego_policy=FLAGS.ego_policy, adv_policy=FLAGS.adv_policy,
                r_ego=FLAGS.r_ego, r_adv=FLAGS.r_adv, sim_seed=FLAGS.seed, gui=False)
    eval_sampler = TrajSampler(env, rootsavepath='None', max_traj_length=FLAGS.max_traj_length)
    
    # load trained re2h2o model
    map_location = {
    'cuda:0': FLAGS.device,
    'cuda:1': FLAGS.device,
    'cuda:2': FLAGS.device,
    'cuda:3': FLAGS.device
    }
    model_adv_re2h2o_policy = torch.load('models_re2h2o_bv//BV0_bv=5.pkl', map_location=map_location)
    sampler_adv_re2h2o_policy = SamplerPolicy(model_adv_re2h2o_policy, device=FLAGS.device)
    
    for l in range(FLAGS.n_loops):
        for epoch in trange(FLAGS.n_epochs):
            metrics = {}
            
            with Timer() as eval_timer:
                ego_policy = FLAGS.ego_policy
                adv_policy = FLAGS.adv_policy
                eval_sampler.env.ego_policy = ego_policy
                eval_sampler.env.adv_policy = 'RL'
                s_a = sampler_adv_re2h2o_policy
                s_e = None
                trajs, _ = eval_sampler.sample(
                    ego_policy=s_e, adv_policy=s_a,
                    n_trajs=FLAGS.eval_n_trajs, deterministic=True
                )
                Eval(metrics, ego_policy, adv_policy, trajs)
                if FLAGS.used_wandb:
                    wandb_logger.log(metrics)
            
            metrics['eval_time'] = eval_timer()
            metrics['epoch_time'] = eval_timer()

if __name__ == '__main__':
    absl.app.run(main)    