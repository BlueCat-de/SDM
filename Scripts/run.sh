CUDA_VISIBLE_DEVICES=2 python main_SDM.py \
  --device cuda:0 \
  --is_save True \
  --save_model True \
  --used_wandb True \
  --reg_scale 0.2 \
  --n_loops 20 \
  --num_agents 4 \
  --r_adv stackelberg3
