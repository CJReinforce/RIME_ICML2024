#!/bin/bash

run_command() {
    algorithm='pebble'
    unsup_steps=19000

    envname="Hopper_v3"
    sac_lr=0.0005
    num_interact=20000
    feedback=100
    reward_batch=10

    seed=$1
    device="cuda:${gpu_ids[$seed]}"
    
    # SAC
    if [ "$algorithm" == "sac" ]; then
        python train_SAC_with_actual_human_labeller.py --device=$device --env="$envname" --seed=$seed --actor_lr=$sac_lr --critic_lr=$sac_lr --steps=1000000  # >> "./reward_model_log_SAC_env_$envname.txt" 2>&1
        
    # PEBBLE
    elif [ "$algorithm" == "pebble" ]; then
        python train_PEBBLE_with_actual_human_labeller.py --device=$device --env="$envname" --seed="$seed" --actor_lr=$sac_lr --critic_lr=$sac_lr --unsup_steps=$unsup_steps --steps=1000000 --num_interact=$num_interact --max_feedback="$feedback" --reward_batch="$reward_batch" --reward_update=50 # > /dev/null 2>&1
        
    # RIME
    else  # [ "$algorithm" == "rime" ]; then
        python train_RIME_with_actual_human_labeller.py --env="$envname" --seed="$seed" --actor_lr=$sac_lr --critic_lr=$sac_lr --unsup_steps=$unsup_steps --steps=1000000 --num_interact=$num_interact --max_feedback="$feedback" --reward_batch=$reward_batch --reward_update=50 --device="$device" --least_reward_update=30 --threshold_variance='kl' --threshold_alpha=0.5 --threshold_beta_init=3.0 --threshold_beta_min=1.0 # >> "RIME_human_label_env_"$envname"_feedback_"$feedback"_mistake_$eps_mistake.txt" 2>&1
        
    fi
}

declare -A gpu_ids=(
    [12345]=0
    [23451]=1
    [34512]=2
    [45123]=3
    [51234]=4
    [67890]=0
    [68790]=1
    [78906]=2
    [89067]=3
    [90678]=4
)

seeds=(12345)  # 23451 34512 45123 51234)  # 67890 68790 78906 89067 90678)

for seed in "${seeds[@]}"; do
    run_command "$seed" 
done

wait
