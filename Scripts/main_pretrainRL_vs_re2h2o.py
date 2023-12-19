import argparse
import numpy as np
from utils import define_flags_with_default, WandbLogger, get_user_flags, set_random_seed, Timer, prefix_metrics, Eval, Count_tensors
from datetime import datetime
from SimpleSAC.envs import Env
from SimpleSAC.sampler import StepSampler, TrajSampler
from SimpleSAC.replay_buffer import ReplayBuffer, GradReplayBuffer
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
parser.add_argument('--adv_policy', type = str, default = 'sumo', choices = ['RL', 'uniform', 'sumo', 'fvdm'])
parser.add_argument('--num_agents', type = int, default = 5)
parser.add_argument('--r_ego', type = str, default = 'stackelberg', choices = ['r1', 'stackelberg'])
parser.add_argument('--r_adv', type = str, default = 'stackelberg3')
# parser.add_argument('--realdata_path', type = str, default = '../datasets/dataset/r3_dis_25_car_6')
parser.add_argument('--is_save', type = str, default = 'False', choices = ['True', 'False'])
parser.add_argument('--device', type = str, default = 'cuda:0')
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--save_model', type=str, default="False", choices=["True", "False"])
parser.add_argument('--n_adv_policy_update_gap', type = int, default = 5)
parser.add_argument('--n_ego_policy_update_gap', type = int, default = 1)
parser.add_argument('--model_name', type = str, default = 'SPG')
parser.add_argument('--gui', type = str, default = 'False', choices = ['True', 'False'])
parser.add_argument('--policy_arch', type = str, default = '256-256')
parser.add_argument('--qf_arch', type = str, default = '256-256')
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--reg_scale', type = float, default = 0.2)
parser.add_argument('--n_epochs', type = int, default = 100)
parser.add_argument('--n_loops', type = int, default = 20)
parser.add_argument('--n_rollout_steps_per_epoch', type = int, default = 1000)
parser.add_argument('--n_train_step_per_epoch', type = int, default = 500)
parser.add_argument('--pretrain_ego', type = str, default = 'True', choices = ['True', 'False'])
parser.add_argument('--pretrain_loops', type = int, default = 2)
parser.add_argument('--pretrain_epochs', type = int, default = 100)
parser.add_argument('--pretrain_steps', type = int, default = 500)
parser.add_argument('--load_pretrain_ego', type = str, default = 'False', choices = ['True', 'False'])
parser.add_argument('--pretrain_ego_path', type = str, default = '')
parser.add_argument('--reset_rb', type = str, default = 'False', choices = ['True', 'False'])
parser.add_argument('--replay_buffer_size', type = int, default = 100000)
parser.add_argument('--pretrain_replay_buffer_size', type = int, default = 1000000)
parser.add_argument('--is_SN', type = str, default = 'True', choices = ['True', 'False'])
parser.add_argument('--is_LN', type = str, default = '')
parser.add_argument('--use_auto_alpha', default = 'False', type = str, choices = ['True', 'False'])
parser.add_argument('--backup_entropy', default = 'True', type = str, choices = ['True', 'False'])
parser.add_argument('--num_save', type = int, default = 5)
args = parser.parse_args()
args.is_save = True if args.is_save == 'True' else False

args.used_wandb = True if args.used_wandb == 'True' else False
# ipdb.set_trace()
args.save_model = True if args.save_model == 'True' else False
args.gui = True if args.gui == 'True' else False
args.pretrain_ego = True if args.pretrain_ego == 'True' else False
args.load_pretrain_ego = True if args.load_pretrain_ego == 'True' else False
args.reset_rb = True if args.reset_rb == 'True' else False
args.is_SN = True if args.is_SN == 'True' else False
args.use_auto_alpha = True if args.use_auto_alpha == 'True' else False
args.backup_entropy = True if args.backup_entropy == 'True' else False

realdata_paths = os.listdir('../datasets/dataset/')
def extract_last_digit(path):
    # This function extracts the last digit from a string and returns it as an integer.
    return int(path[-1])

# Sort the realdata_paths list based on the last digit in ascending order.
sorted_realdata_paths = sorted(realdata_paths, key=extract_last_digit)
realdata_path = os.path.join('../datasets/dataset', sorted_realdata_paths[args.num_agents - 1])
# ipdb.set_trace()



FLAGS_DEF = define_flags_with_default(
    model_name = args.model_name,
    used_wandb = args.used_wandb,
    ego_policy = args.ego_policy,
    adv_policy = args.adv_policy,
    num_agents = args.num_agents,
    reg_scale = args.reg_scale,
    r_ego = args.r_ego,
    r_adv = args.r_adv,
    r_adv_replaybuffer = args.r_adv,
    realdata_path = realdata_path,
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
    n_ego_policy_update_gap = args.n_ego_policy_update_gap,
    eval_period = 10,
    eval_n_trajs = 20,
    logging = WandbLogger.get_default_config(),
    gui = args.gui,
    pretrain_ego = args.pretrain_ego,
    pretrain_epochs = args.pretrain_epochs,
    pretrain_loops = args.pretrain_loops,
    pretrain_steps = args.pretrain_steps,
    load_pretrain_ego = args.load_pretrain_ego,
    pretrain_ego_path = args.pretrain_ego_path,
    reset_rb = args.reset_rb,
    cql_ego = SAC.get_default_config(),
    is_SN = args.is_SN,
    is_LN = args.is_LN,
    use_auto_alpha = args.use_auto_alpha,
    backup_entropy = args.backup_entropy,
    num_save = args.num_save
)

def argparse():
    ...

def get_tensors_on_gpu(device):
    tensors_on_gpu = []
    for obj in dir():
        if isinstance(eval(obj), torch.Tensor):
            if eval(obj).device == device:
                tensors_on_gpu.append(obj)
    return tensors_on_gpu

def main(argv):
    
    # ipdb.set_trace()
    FLAGS = absl.flags.FLAGS
    if FLAGS.is_save:
        eval_savepath = "output/" + \
                        f"bv={FLAGS.num_agents}-{FLAGS.adv_policy}_"  \
                        f"r-ego={FLAGS.r_ego}_r-adv={FLAGS.r_adv}_" \
                        f"seed={FLAGS.seed}_time={FLAGS.current_time}" \
                        f"reg_scale={FLAGS.reg_scale}_" \
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
        wandb_logger = WandbLogger(config=FLAGS.logging, variant=variant, seed = FLAGS.seed)
        wandb.run.name = f"{FLAGS.model_name}" \
                         f"_Pretrain_Train_Eval_{FLAGS.model_name}" \
                         f"bv={FLAGS.num_agents}-{FLAGS.adv_policy}_" \
                         f"r-ego={FLAGS.r_ego}_r-adv={FLAGS.r_adv}_" \
                         f"reg_scale={FLAGS.reg_scale}_" \
                         f"pretrain_adv={FLAGS.adv_policy}_" \
                         f"pretrain_epochs={FLAGS.pretrain_epochs}_" \
                         f"pretrain_steps={FLAGS.pretrain_steps}_" \
                         f"pretrain_rb_size={FLAGS.pretrain_replay_buffer_size}_" \
                         f"reset_rb={FLAGS.reset_rb}_" \
                         f"n_adv_policy_update_gap={FLAGS.n_adv_policy_update_gap}_" \
                         f"n_ego_policy_update_gap={FLAGS.n_ego_policy_update_gap}_" \
                         f"is_SN={FLAGS.is_SN}_" \
                         f"is_auto_alpha={FLAGS.use_auto_alpha}_" \
                         f"backup_entropy={FLAGS.backup_entropy}_" \
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
    replay_buffer = GradReplayBuffer(num_state, num_action_ego, num_action_adv, FLAGS.replay_buffer_size, device=FLAGS.device) \
        if FLAGS.model_name == 'SPG' else ReplayBuffer(num_state, num_action_ego, num_action_adv, FLAGS.replay_buffer_size, device=FLAGS.device)

    # ipdb.set_trace()
    


    
    ego_policy = TanhGaussianPolicy(
        num_state,
        num_action_ego,
        arch=FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
        orthogonal_init=FLAGS.orthogonal_init,
        is_SN = False,
        is_LN = FLAGS.is_LN
    )
    qf1_ego = FullyConnectedQFunction(
        num_state,
        num_action_ego,
        num_action_adv,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
        is_LN=FLAGS.is_LN,
        is_SN=False
    )
    target_qf1_ego = deepcopy(qf1_ego)
    qf2_ego = FullyConnectedQFunction(
        num_state,
        num_action_ego,
        num_action_adv,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
        is_LN=FLAGS.is_LN,
        is_SN=False
    )
    target_qf2_ego = deepcopy(qf2_ego)
    sampler_ego_policy = SamplerPolicy(ego_policy, FLAGS.device)     


    
    adv_policy = TanhGaussianPolicy(
        num_state,
        num_action_adv,
        arch=FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
        orthogonal_init=FLAGS.orthogonal_init,
        is_LN=FLAGS.is_LN,
        is_SN=FLAGS.is_SN
    )
    qf1_adv = FullyConnectedQFunction(
        num_state,
        num_action_ego,
        num_action_adv,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
        is_LN=FLAGS.is_LN,
        is_SN=FLAGS.is_SN
    )
    target_qf1_adv = deepcopy(qf1_adv)
    qf2_adv = FullyConnectedQFunction(
        num_state,
        num_action_ego,
        num_action_adv,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
        is_LN=FLAGS.is_LN,
        is_SN=FLAGS.is_SN
    )
    target_qf2_adv = deepcopy(qf1_adv)
    sampler_adv_policy = SamplerPolicy(adv_policy, FLAGS.device)

    if FLAGS.model_name == 'SPG':
        model = SDM(None, 
                       ego_policy = ego_policy, 
                       adv_policy = adv_policy, 
                       qf1_ego = qf1_ego,
                       qf2_ego = qf2_ego,
                       target_qf1_ego = target_qf1_ego,
                       target_qf2_ego = target_qf2_ego,
                       qf1_adv = qf1_adv,
                       qf2_adv = qf2_adv,
                       target_qf1_adv = target_qf1_adv,
                       target_qf2_adv = target_qf2_adv,
                       device = FLAGS.device,
                       reg_scale = FLAGS.reg_scale,
                       use_automatic_entropy_tuning = FLAGS.use_auto_alpha,
                       backup_entropy = FLAGS.backup_entropy,)
    else:
        return
    model.torch_to_device(FLAGS.device)

    if FLAGS.pretrain_ego:
        if not FLAGS.load_pretrain_ego:
            model_pre_ego = SAC(
                FLAGS.cql_ego,
                policy = ego_policy,
                qf1 = qf1_ego,
                qf2 = qf2_ego,
                target_qf1 = target_qf1_ego,
                target_qf2 = target_qf2_ego
            )
            model_pre_ego.torch_to_device(FLAGS.device)
        else:
            model_pre_ego = torch.load(FLAGS.pretrain_ego_path)

    viskit_metrics = {}

    # TODO: Pretrain Ego Policy on fvdm BV using sac
    # TODO: Check bv
    # TODO: Check sac
    if FLAGS.pretrain_ego:
        pretrain_replay_buffer.reset()
        sampler_pretrain_ego_policy = SamplerPolicy(model_pre_ego.policy, FLAGS.device)
        sampler_pretrain_ego_policy.set_grad(False)
        
        for i in range(FLAGS.pretrain_loops):
            
            for epoch in trange(FLAGS.pretrain_epochs):
                metrics = {}
                pretrain_sampler.env.adv_policy = FLAGS.adv_policy # sumo
                pretrain_sampler.env.ego_policy = 'RL'
                # while True:
                pretrain_sampler.sample(
                    ego_policy=sampler_pretrain_ego_policy, adv_policy=None, n_steps=FLAGS.n_rollout_steps_per_epoch,
                    deterministic=False, replay_buffer=pretrain_replay_buffer,
                    joint_noise_std=FLAGS.joint_noise_std
                )
                metrics['epoch'] = epoch
                for batch_idx in trange(FLAGS.pretrain_steps):
                    batch = pretrain_replay_buffer.sample(FLAGS.batch_size)
                    # if FLAGS.used_wandb:
                    #     wandb_logger.log(prefix_metrics(model_pre_ego.train(batch), 'SAC_Pretrain'))
                    metrics.update(prefix_metrics(model_pre_ego.train(batch), 'SAC_Pretrain'))
                    
                if FLAGS.used_wandb:
                    wandb_logger.log(metrics)
                # eval
                
                if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                    eval_ego_policy = 'RL'
                    eval_sampler.env.ego_policy = eval_ego_policy
                    eval_sampler.env.adv_policy = FLAGS.adv_policy
                    if adv_policy != 'RL':
                        s_a = None
                    else:
                        s_a = sampler_adv_policy
                    # ipdb.set_trace()
                    trajs, _ = eval_sampler.sample(
                        ego_policy=sampler_pretrain_ego_policy, adv_policy=s_a,
                        n_trajs=FLAGS.eval_n_trajs, deterministic=True
                    )
                    # TODO: add speed
                    Eval(metrics, eval_ego_policy, FLAGS.adv_policy, trajs)
                    
                    if FLAGS.used_wandb:
                        wandb_logger.log(metrics)  
                        
        # pretrain_replay_buffer.reset()
            
        if FLAGS.save_model:
            torch.save(model_pre_ego, os.path.join(eval_savepath, 'models', 'pretrain_ego.pth'))
            if FLAGS.used_wandb:
                # wandb_logger.log(metrics)
                pre_save_data = {'model_pre_ego': model_pre_ego}
                wandb_logger.save_pickle(pre_save_data, 'pre_model.pkl')
        sampler_pretrain_ego_policy = deepcopy(sampler_pretrain_ego_policy) # freezing the pretrain policy
    # return
    #  = 1
    # ipdb.set_trace()
    
    # load trained re2h2o model
    map_location = {
    'cuda:0': FLAGS.device,
    'cuda:1': FLAGS.device,
    'cuda:2': FLAGS.device,
    'cuda:3': FLAGS.device
    }
    # model_adv_re2h2o_policy = torch.load('models_re2h2o_bv//BV0_bv=1.pkl', map_location=map_location)
    model_adv_re2h2o_policy = torch.load(f'models_re2h2o_bv//BV0_bv={FLAGS.num_agents}.pkl', map_location=map_location)
    sampler_adv_re2h2o_policy = SamplerPolicy(model_adv_re2h2o_policy, device=FLAGS.device)
    
    for l in range(FLAGS.n_loops):
        for epoch in trange(FLAGS.n_epochs):

            '''leader and follower'''
            metrics = {}
 
            # TODO: Evaluate in the real world
            with Timer() as eval_timer:
                if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                    # Eval
                    eval_sampler.env.ego_policy = 'RL'
                    eval_sampler.env.adv_policy = 'RL'
                    s_e = sampler_pretrain_ego_policy
                    ego_policy = 'pretrainedRL'
                    adv_policy = 're2h2o'
                    s_a = sampler_adv_re2h2o_policy
                    trajs, _ = eval_sampler.sample(
                            ego_policy=s_e, adv_policy=s_a,
                            n_trajs=FLAGS.eval_n_trajs, deterministic=True
                        )
                    Eval(metrics, ego_policy, adv_policy, trajs)
                    
                    if FLAGS.used_wandb:
                        wandb_logger.log(metrics)
               
            # metrics['rollout_time'] = rollout_timer()
            # metrics['train_time'] = train_timer()
            metrics['eval_time'] = eval_timer()
            # metrics['epoch_time'] = train_timer() + eval_timer()
            if FLAGS.used_wandb:
                save_data = {FLAGS.model_name: model,
                        'variant': variant, 'epoch': epoch}
                wandb_logger.save_pickle(save_data, 'model.pkl')
            viskit_metrics.update(metrics)
            logger.record_dict(viskit_metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)

        # save model for matric Eval
        # if FLAGS.save_model and l % (FLAGS.n_loops / FLAGS.num_save) == 0 or l == FLAGS.n_loops - 1:
        #     # ipdb.set_trace()
        #     torch.save(model, os.path.join(eval_savepath, 'models', f'loop_{l+1}.pth'))
                

    if FLAGS.save_model and FLAGS.used_wandb:
        save_data = {FLAGS.model_name: model,
                    'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')
    # if FLAGS.save_model:
    #     torch.save(model, os.path.join(eval_savepath, 'models', 'trained_model.pth'))

if __name__ == '__main__':
    absl.app.run(main)
    

    


