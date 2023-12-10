#!/bin/bash

# task func
run_task() {
    num_agents=$1
    reg_scale=$2
    seed=$3
    n_adv_policy_update_gap=$4
    n_ego_policy_update_gap=$5
    device=$6

    # command
    command="taskset -c 65-256 python main_SDM.py --device $device --used_wandb True --is_save True --save_model True --num_agents $num_agents --reg_scale $reg_scale --seed $seed --r_adv stackelberg3 --n_adv_policy_update_gap $n_adv_policy_update_gap --n_ego_policy_update_gap $n_ego_policy_update_gap"
    # run command
    $command
}

# params range
num_agents_values=(1 2 3 4 5)
reg_scale_values=(0 0.2 1 2 10)
seed_values=(127)
n_adv_policy_update_gap_values=(1 5)
n_ego_policy_update_gap_values=(1 5)

# max GPU num
num_gpus=4

# max process num
max_processes=100
current_processes=0

# traverse all params
for num_agents in "${num_agents_values[@]}"; do
    for reg_scale in "${reg_scale_values[@]}"; do
        for seed in "${seed_values[@]}"; do
            for n_adv_policy_update_gap in "${n_adv_policy_update_gap_values[@]}"; do
                for n_ego_policy_update_gap in "${n_ego_policy_update_gap_values[@]}"; do
                    # filter out
                    if [ $n_ego_policy_update_gap != 5 ] || [ $n_adv_policy_update_gap != 5 ]; then
                        if ([ $reg_scale == 1 ]) || \
                            ([ $reg_scale != 1 ] && [ $n_ego_policy_update_gap == 1 ] && [ $n_adv_policy_update_gap == 5 ]); then
                            # GPU allocate
                            device=$(($current_processes % $num_gpus))

                            # start
                            run_task $num_agents $reg_scale $seed $n_adv_policy_update_gap $n_ego_policy_update_gap "cuda:$device" &
                            ((current_processes++))

                            # control process num
                            if ((current_processes >= max_processes)); then
                                wait
                                current_processes=0
                            fi
                        fi
                    fi
                done
            done
        done
    done
done

# wait for all to finish
wait