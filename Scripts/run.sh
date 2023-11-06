CUDA_VISIBLE_DEVICES=1 python main_SDM.py \
  --device cuda:0 \
  --is_save True \
  --save_model True \
  --used_wandb True \
  --reg_scale 1 \
  --n_loops 20 \
  --num_agents 1 \
  --realdata_path '../datasets/dataset/r3_dis_10_car_2' \
  --r_adv stackelberg3
