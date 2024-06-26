#!/bin/bash

run_command() {
    algorithm='pebble'  # [pebble, rime]
    unsup_steps=19000

    envname="Hopper_v3"
    sac_lr=0.0005
    num_interact=20000
    feedback=100
    reward_batch=10

    seed=$1
    device="cuda:${gpu_ids[$seed]}"

    # PEBBLE
    if [ "${algorithm,,}" == "pebble" ]; then
        python train_PEBBLE_with_actual_human_labeller.py --device=$device --env="$envname" --seed="$seed" --actor_lr=$sac_lr --critic_lr=$sac_lr --unsup_steps=$unsup_steps --steps=10000000 --num_interact=$num_interact --max_feedback="$feedback" --reward_batch="$reward_batch" --reward_update=50
        
    # RIME
    else
        python train_RIME_with_actual_human_labeller.py --env="$envname" --seed="$seed" --actor_lr=$sac_lr --critic_lr=$sac_lr --unsup_steps=$unsup_steps --steps=10000000 --num_interact=$num_interact --max_feedback="$feedback" --reward_batch=$reward_batch --reward_update=50 --device="$device" --least_reward_update=30 --threshold_variance='kl' --threshold_alpha=0.5 --threshold_beta_init=3.0 --threshold_beta_min=1.0
        
    fi
}

declare -A gpu_ids=(
    [12345]=0
)

seeds=(12345)

for seed in "${seeds[@]}"; do
    run_command "$seed" 
done

wait
