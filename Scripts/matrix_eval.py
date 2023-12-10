import argparse
import numpy as np
from utils import define_flags_with_default, WandbLogger, get_user_flags, set_random_seed, Timer, prefix_metrics, Eval
from datetime import datetime
from SimpleSAC.envs import Env
from SimpleSAC.sampler import StepSampler, TrajSampler
from SimpleSAC.replay_buffer import ReplayBuffer, GradReplayBuffer
from SimpleSAC.mixed_replay_buffer import MixedReplayBuffer
from SimpleSAC.sac import SAC
from SimpleSAC.models.model import TanhGaussianPolicy, SamplerPolicy, FullyConnectedQFunction

from SDM.SDM import SDM

from copy import deepcopy
import os
import absl.app
import absl.flags
import wandb
from viskit.logging import logger, setup_logger
from tqdm import trange
import ipdb
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--used_wandb', type = str, default = 'False', choices = ['True', 'False'])
parser.add_argument('--ego_policy', type = str, default = 'RL', choices = ['RL', 'uniform', 'sumo', 'fvdm'])
parser.add_argument('--adv_policy', type = str, default = 'RL', choices = ['RL', 'uniform', 'sumo', 'fvdm'])
parser.add_argument('--num_agents', type = int, default = 5)
parser.add_argument('--r_ego', type = str, default = 'r1', choices = ['r1', 'stackelberg'])
parser.add_argument('--r_adv', type = str, default = 'r1')
parser.add_argument('--realdata_path', type = str, default = '/home/qh802/cqm/Cross-Learning/datasets/dataset/r3_dis_25_car_6')
parser.add_argument('--is_save', type = str, default = 'False', choices = ['True', 'False'])
parser.add_argument('--device', type = str, default = 'cuda:0')
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--save_model', type=str, default="True", choices=["True", "False"])
parser.add_argument('--n_adv_policy_update_gap', type = int, default = 4)
parser.add_argument('--model_name', type = str, default = 'SPG')
parser.add_argument('--gui', type = str, default = 'False', choices = ['True', 'False'])
parser.add_argument('--policy_arch', type = str, default = '256-256')
parser.add_argument('--qf_arch', type = str, default = '256-256')
parser.add_argument('--batch_size', type = int, default = 256)
parser.add_argument('--reg_reward', type = str, default = 'True', choices = ['True', 'False'])
parser.add_argument('--n_epochs', type = int, default = 2)
parser.add_argument('--n_loops', type = int, default = 10)
parser.add_argument('--n_rollout_steps_per_epoch', type = int, default = 1000)
parser.add_argument('--n_train_step_per_epoch', type = int, default = 100)
parser.add_argument('--pretrain_ego', type = str, default = 'False', choices = ['True', 'False'])
parser.add_argument('--pretrain_loop', type = int, default = 1)
parser.add_argument('--pretrain_epochs', type = int, default = 100)
parser.add_argument('--pretrain_steps', type = int, default = 1000)
parser.add_argument('--load_pretrain_ego', type = str, default = 'False', choices = ['True', 'False'])
parser.add_argument('--pretrain_ego_path', type = str, default = '')
parser.add_argument('--train_model_path', type = str, default = '/home/qh802/cqm/Cross-Learning/Scripts/output/bv=5-sumo_r-ego=stackelberg_r-adv=stackelberg3_seed=42_time=2023-08-29-22-37-49reg_scale=0.2_pretrain_adv=sumo/models')
parser.add_argument('--reset_rb', type = str, default = 'False', choices = ['True', 'False'])
parser.add_argument('--replay_buffer_size', type = int, default = 100000)
parser.add_argument('--pretrain_replay_buffer_size', type = int, default = 1000000)
args = parser.parse_args()
args.is_save = True if args.is_save == 'True' else False
args.used_wandb = True if args.used_wandb == 'True' else False
args.save_model = True if args.save_model == 'True' else False
args.reg_reward = True if args.reg_reward == 'True' else False
args.gui = True if args.gui == 'True' else False
args.pretrain_ego = True if args.pretrain_ego == 'True' else False
args.load_pretrain_ego = True if args.load_pretrain_ego == 'True' else False
args.reset_rb = True if args.reset_rb == 'True' else False


FLAGS_DEF = define_flags_with_default(
    model_name = args.model_name,
    used_wandb = args.used_wandb,
    ego_policy = args.ego_policy,
    adv_policy = args.adv_policy,
    num_agents = args.num_agents,
    reg_reward = args.reg_reward,
    r_ego = args.r_ego,
    r_adv = args.r_adv,
    r_adv_replaybuffer = args.r_adv,
    realdata_path = args.realdata_path,
    is_save = args.is_save,
    device = args.device,
    seed = args.seed,
    replay_buffer_size = args.replay_buffer_size,
    pretrain_replay_buffer_size = args.pretrain_replay_buffer_size,
    save_model = args.save_model,
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    replaybuffer_ratio = 10,
    real_residual_ratio = 1.0,
    dis_dropout = False,
    max_traj_length = 100,
    batch_size = args.batch_size,
    reward_scale = 1.0,
    reward_bias = 0.0,
    clip_action = 1.0,
    joint_noise_std = 0.0,
    policy_arch = args.policy_arch,
    qf_arch = args.qf_arch,
    orthogonal_init = False,
    policy_log_std_multiplier = 1.0,
    policy_log_std_offset = -1.0,
    # train and evaluate policy
    n_epochs = args.n_epochs,
    n_loops = args.n_loops,
    bc_epochs = 0,
    n_rollout_steps_per_epoch = args.n_rollout_steps_per_epoch,
    n_train_step_per_epoch = args.n_train_step_per_epoch,
    n_adv_policy_update_gap = args.n_adv_policy_update_gap,
    eval_period = 1,
    eval_n_trajs = 20,
    logging = WandbLogger.get_default_config(),
    gui = args.gui,
    pretrain_ego = args.pretrain_ego,
    pretrain_epochs = args.pretrain_epochs,
    pretrain_loop = args.pretrain_loop,
    pretrain_steps = args.pretrain_steps,
    load_pretrain_ego = args.load_pretrain_ego,
    pretrain_ego_path = args.pretrain_ego_path,
    train_model_path = args.train_model_path,
    reset_rb = args.reset_rb,
    cql_ego = SAC.get_default_config()
)

def argparse():
    ...

def main(argv):
    FLAGS = absl.flags.FLAGS
    if FLAGS.is_save:
        eval_savepath = "output/" + \
                        f"bv={FLAGS.num_agents}-{FLAGS.adv_policy}_" \
                        f"r-ego={FLAGS.r_ego}_r-adv={FLAGS.r_adv}_" \
                        f"seed={FLAGS.seed}_time={FLAGS.current_time}" \
                        f"reg_reward={FLAGS.reg_reward}_" \
                        f"pretrain_adv={FLAGS.adv_policy}" \
                        + "/"
        if not os.path.exists('output'):
            os.makedirs('output')
        if not os.path.exists(eval_savepath):
            os.mkdir(eval_savepath)
            os.mkdir(eval_savepath + "avcrash")
            os.mkdir(eval_savepath + "bvcrash")
            os.mkdir(eval_savepath + "avarrive")
            os.mkdir(eval_savepath + "models")
    else:
        eval_savepath = 'None'

    if FLAGS.used_wandb:
        variant = get_user_flags(FLAGS, FLAGS_DEF)
        wandb_logger = WandbLogger(config=FLAGS.logging, variant=variant)
        wandb.run.name = f"StackAC_EVAL_{FLAGS.model_name}" \
                         f"bv={FLAGS.num_agents}-{FLAGS.adv_policy}_" \
                         f"r-ego={FLAGS.r_ego}_r-adv={FLAGS.r_adv}_" \
                         f"reg_reward={FLAGS.reg_reward}_" \
                         f"pretrain_adv={FLAGS.adv_policy}_" \
                         f"pretrain_epochs={FLAGS.pretrain_epochs}_" \
                         f"pretrain_steps={FLAGS.pretrain_steps}_" \
                         f"pretrain_rb_size={FLAGS.pretrain_replay_buffer_size}_" \
                         f"seed={FLAGS.seed}_time={FLAGS.current_time}"
        setup_logger(
            variant=variant,
            exp_id=wandb_logger.experiment_id,
            seed=FLAGS.seed,
            base_log_dir=FLAGS.logging.output_dir,
            include_exp_prefix_sub_dir=False
        )

    set_random_seed(FLAGS.seed)
    # real_env = Env(realdata_path=FLAGS.realdata_path, num_agents=FLAGS.num_agents, sim_horizon=FLAGS.max_traj_length,
    #                ego_policy=FLAGS.ego_policy, adv_policy=FLAGS.adv_policy,
    #                r_ego=FLAGS.r_ego, r_adv=FLAGS.r_adv, sim_seed=FLAGS.seed, gui=FLAGS.gui)
    env = Env(realdata_path=FLAGS.realdata_path, num_agents=FLAGS.num_agents, sim_horizon=FLAGS.max_traj_length,
                   ego_policy=FLAGS.ego_policy, adv_policy=FLAGS.adv_policy,
                   r_ego=FLAGS.r_ego, r_adv=FLAGS.r_adv, sim_seed=FLAGS.seed, gui=FLAGS.gui)
    pretain_env = Env(realdata_path=FLAGS.realdata_path, num_agents=FLAGS.num_agents, sim_horizon=FLAGS.max_traj_length,
                      ego_policy=FLAGS.ego_policy, adv_policy=FLAGS.adv_policy,
                      r_ego = 'r1', r_adv = 'r3', sim_seed=FLAGS.seed, gui=FLAGS.gui)
    pretrain_sampler = StepSampler(pretain_env, max_traj_length=FLAGS.max_traj_length)
    train_sampler = StepSampler(env, max_traj_length=FLAGS.max_traj_length)
    eval_sampler = TrajSampler(env, rootsavepath=eval_savepath, max_traj_length=FLAGS.max_traj_length)
    
    # replay buffer
    num_state = env.state_space[0]
    num_action_adv = env.action_space_adv[0]
    num_action_ego = env.action_space_ego[0]
    # ipdb.set_trace()
    num_action = num_action_ego + num_action_adv
    pretrain_replay_buffer = ReplayBuffer(num_state, num_action_ego, num_action_adv, FLAGS.pretrain_replay_buffer_size, device=FLAGS.device) 
    replay_buffer = GradReplayBuffer(num_state, num_action_ego, num_action_adv, FLAGS.replay_buffer_size, device=FLAGS.device) 

    # ipdb.set_trace()
    


    viskit_metrics = {}
    model_names = os.listdir(FLAGS.train_model_path)
    model_names.remove('best_av_model.pth')
    model_names.remove('pretrain_ego.pth')
    sorted_list = sorted(model_names, key=lambda x: int(x.split('_')[1].split('.')[0]))
    model_lists = []
    sampler_ego_policy_list = []
    sampler_adv_policy_list = []
    for model_name in sorted_list:
        # ipdb.set_trace()
        if 'loop' not in model_name:
            continue
        model = torch.load(os.path.join(FLAGS.train_model_path, model_name))
        model_lists.append(model)
    for model in model_lists:
        sampler_ego_policy = SamplerPolicy(model.ego_policy, FLAGS.device)
        sampler_adv_policy = SamplerPolicy(model.adv_policy, FLAGS.device)
        sampler_ego_policy_list.append(sampler_ego_policy)
        sampler_adv_policy_list.append(sampler_adv_policy)
    pretrain_model = torch.load(os.path.join(FLAGS.train_model_path, 'pretrain_ego.pth'))
    sampler_ego_policy = SamplerPolicy(pretrain_model.policy, FLAGS.device)
    sampler_ego_policy_list.append(sampler_ego_policy) # add pretrain ego policy
    # TODO: Pretrain Ego Policy on fvdm BV using sac
    # TODO: Check bv
    # TODO: Check sac
    for i in range(len(sampler_ego_policy_list)):
        for j in range(len(sampler_adv_policy_list)):
            sampler_ego_policy = sampler_ego_policy_list[i]
            sampler_adv_policy = sampler_adv_policy_list[j]
            for l in range(FLAGS.n_loops):
                for epoch in trange(FLAGS.n_epochs):

                    '''leader and follower'''
                    metrics = {}

                    metrics['epoch'] = epoch
                    
                    # TODO: Evaluate in the real world
                    with Timer() as eval_timer:
                        # Eval ego policy

                        eval_ego_policy = 'RL'
                        eval_adv_policy = 'RL'
                        eval_sampler.env.ego_policy = eval_ego_policy
                        eval_sampler.env.adv_policy = eval_adv_policy
                        # ipdb.set_trace()
                        trajs = eval_sampler.sample(
                            ego_policy=sampler_ego_policy, adv_policy=sampler_adv_policy,
                            n_trajs=FLAGS.eval_n_trajs, deterministic=True
                        )
                        Eval(metrics, f'{eval_ego_policy}{i}', f'{eval_adv_policy}{j}', trajs)

                    if FLAGS.used_wandb:
                        save_data = {FLAGS.model_name: model,
                                    'variant': variant, 'epoch': epoch}
                        wandb_logger.save_pickle(save_data, 'model.pkl')
                    # metrics['rollout_time'] = rollout_timer()
                    metrics['eval_time'] = eval_timer()
                    metrics['epoch_time'] = eval_timer()
                    if FLAGS.used_wandb:
                            wandb_logger.log(metrics)
                    viskit_metrics.update(metrics)
                    logger.record_dict(viskit_metrics)
                    logger.dump_tabular(with_prefix=False, with_timestamp=False)
        


            


if __name__ == '__main__':
    absl.app.run(main)
    

    


