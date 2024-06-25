#!/bin/bash

run_command() {
    eps_mistake=0.3
    eps_skip=0.0
    eps_equal=0.0
    teacher_gamma=1.0
    algorithm='pebble'

    # parameters for algorithms
    # PEBBLE (robust algorithm): ['', 'MAE', 'tCE', 'ADT', 'LS']
    robust_algo=''
    # RIME
    unsup_steps=9000  # 2000 for cheetah_run otherwise 9000
    # MRN
    meta_steps=5000
    # RUNE
    rho=0.00001
    # SURF
    tau=0.99

    # envname="Hopper_v3"
    # sac_lr=0.0005
    # num_interact=10000
    # feedback=500
    # reward_batch=100

    envname="walker_walk"
    sac_lr=0.0005
    num_interact=20000
    feedback=1000
    reward_batch=50

    # envname="cheetah_run"
    # sac_lr=0.0005
    # num_interact=20000
    # feedback=10000
    # reward_batch=500

    # envname="quadruped_walk"
    # sac_lr=0.0001
    # num_interact=30000
    # feedback=2000
    # reward_batch=200

    # envname="metaworld_button-press-v2"
    # sac_lr=0.0003
    # num_interact=5000
    # feedback=20000
    # reward_batch=100

    # envname="metaworld_sweep-into-v2"
    # sac_lr=0.0003
    # num_interact=5000
    # feedback=20000
    # reward_batch=100

    # envname="metaworld_drawer-open-v2"
    # sac_lr=0.0003
    # num_interact=5000
    # feedback=10000
    # reward_batch=100

    # envname="metaworld_hammer-v2"
    # sac_lr=0.0003
    # num_interact=5000
    # feedback=20000
    # reward_batch=100

    seed=$1
    device="cuda:${gpu_ids[$seed]}"
    
    # SAC
    if [ "$algorithm" == "sac" ]; then
        case "$envname" in
        *metaworld*)
            python train_SAC.py --device=$device --env="$envname" --seed=$seed --actor_lr=$sac_lr --critic_lr=$sac_lr --steps=1000000 --batch_size=512 --critic_hidden_dim=256 --critic_hidden_depth=3 --actor_hidden_dim=256 --actor_hidden_depth=3 >> "./reward_model_log_SAC_env_"$envname".txt" 2>&1
            ;;
        *)
            python train_SAC.py --device=$device --env="$envname" --seed=$seed --actor_lr=$sac_lr --critic_lr=$sac_lr --steps=1000000  >> "./reward_model_log_SAC_env_$envname.txt" 2>&1
            ;;
        esac

    # MRN
    elif [ "$algorithm" == "mrn" ]; then
        case "$envname" in
        *metaworld*)
            python train_MRN.py --env="$envname" --seed=$seed --actor_lr=$sac_lr --critic_lr=$sac_lr  --batch_size=512 --critic_hidden_dim=256 --critic_hidden_depth=3 --actor_hidden_dim=256 --actor_hidden_depth=3 --unsup_steps=9000 --steps=1000000 --num_interact=$num_interact --max_feedback="$feedback" --reward_batch=$reward_batch --reward_update=10 --feed_type=1 --meta_steps=$meta_steps --device=$device --eps_mistake="$eps_mistake" >> "./reward_model_log_MRN_env_"$envname"_feedback_"$feedback"_mistake_$eps_mistake.txt" 2>&1
            ;;
        *)
            python train_MRN.py --env="$envname" --seed=$seed --actor_lr=$sac_lr --critic_lr=$sac_lr --unsup_steps=9000 --steps=1000000 --num_interact=$num_interact --max_feedback="$feedback" --reward_batch=$reward_batch --reward_update=50 --feed_type=1 --meta_steps=$meta_steps --device=$device --eps_mistake="$eps_mistake" >> "./reward_model_log_MRN_env_"$envname"_feedback_"$feedback"_mistake_$eps_mistake.txt" 2>&1
            ;;
        esac

    # PEBBLE
    elif [ "$algorithm" == "pebble" ]; then
        case "$envname" in
        *metaworld*)
            python train_PEBBLE"$robust_algo".py --device=$device --env="$envname" --seed="$seed" --actor_lr=$sac_lr --critic_lr=$sac_lr --unsup_steps=9000 --steps=1000000 --batch_size=512 --critic_hidden_dim=256 --critic_hidden_depth=3 --actor_hidden_dim=256 --actor_hidden_depth=3 --num_interact=$num_interact --max_feedback="$feedback" --reward_batch="$reward_batch" --reward_update=10 --feed_type=1 --eps_mistake="$eps_mistake" --eps_skip="$eps_skip" --eps_equal="$eps_equal" --teacher_gamma="$teacher_gamma" > /dev/null 2>&1
            ;;
        *)
            python train_PEBBLE"$robust_algo".py --device=$device --env="$envname" --seed="$seed" --actor_lr=$sac_lr --critic_lr=$sac_lr --unsup_steps=9000 --steps=1000000 --num_interact=$num_interact --max_feedback="$feedback" --reward_batch="$reward_batch" --reward_update=50 --feed_type=1 --eps_mistake="$eps_mistake" --eps_skip="$eps_skip" --eps_equal="$eps_equal" --teacher_gamma="$teacher_gamma" > /dev/null 2>&1
            ;;
            
        esac

    # RIME
    elif [ "$algorithm" == "rime" ]; then
        case "$envname" in
        *metaworld*)
            python train_RIME.py --device=$device --env="$envname" --seed="$seed" --actor_lr=$sac_lr --critic_lr=$sac_lr --unsup_steps=9000 --steps=1000000 --batch_size=512 --critic_hidden_dim=256 --critic_hidden_depth=3 --actor_hidden_dim=256 --actor_hidden_depth=3 --num_interact=$num_interact --max_feedback="$feedback" --reward_batch="$reward_batch" --reward_update=10 --feed_type=1 --eps_mistake="$eps_mistake" --least_reward_update=5 --threshold_variance='kl' --threshold_alpha=0.5 --threshold_beta_init=3.0 --threshold_beta_min=1.0 --eps_skip="$eps_skip" --eps_equal="$eps_equal" --teacher_gamma="$teacher_gamma" >> "./log_RIME_env_"$envname"_feedback_"$feedback"_mistake_$eps_mistake.txt" 2>&1
            ;;
        *)
            python train_RIME.py --env="$envname" --seed="$seed" --actor_lr=$sac_lr --critic_lr=$sac_lr --unsup_steps=$unsup_steps --steps=1000000 --num_interact=$num_interact --max_feedback="$feedback" --reward_batch=$reward_batch --reward_update=50 --feed_type=1 --device="$device" --eps_mistake="$eps_mistake" --least_reward_update=15 --threshold_variance='kl' --threshold_alpha=0.5 --threshold_beta_init=3.0 --threshold_beta_min=1.0 --eps_skip="$eps_skip" --eps_equal="$eps_equal" --teacher_gamma="$teacher_gamma"  #  >> "./log_RIME_env_"$envname"_feedback_"$feedback"_mistake_$eps_mistake.txt" 2>&1
            ;;
        esac

    # SURF
    elif [ "$algorithm" == "surf" ]; then
        case "$envname" in
        *metaworld*)
            python train_SURF.py --device=$device --env="$envname" --seed="$seed" --actor_lr=$sac_lr --critic_lr=$sac_lr --unsup_steps=9000 --steps=1000000 --batch_size=512 --critic_hidden_dim=256 --critic_hidden_depth=3 --actor_hidden_dim=256 --actor_hidden_depth=3 --num_interact=$num_interact --max_feedback="$feedback" --reward_batch="$reward_batch" --reward_update=20 --feed_type=1 --eps_mistake="$eps_mistake" --inv_label_ratio=10 --threshold_u=$tau --mu=4 >> "./reward_model_log_SURF_env_"$envname"_feedback_"$feedback"_mistake_$eps_mistake.txt" 2>&1
            ;;
        *)
            python train_SURF.py --device=$device --env="$envname" --seed=$seed --actor_lr=$sac_lr --critic_lr=$sac_lr --unsup_steps=9000 --steps=1000000 --num_interact=$num_interact --max_feedback="$feedback" --reward_batch=$reward_batch --inv_label_ratio=20 --reward_update=100 --feed_type=1 --eps_mistake="$eps_mistake" --threshold_u=$tau --mu=4 >> "./reward_model_log_SURF_env_"$envname"_feedback_"$feedback"_mistake_$eps_mistake.txt" 2>&1
            ;;
        esac

    # RUNE
    else
        case "$envname" in
        *metaworld*)
            python train_RUNE.py --device=$device --env="$envname" --seed="$seed" --actor_lr=$sac_lr --critic_lr=$sac_lr --unsup_steps=9000 --steps=1000000 --batch_size=512 --critic_hidden_dim=256 --critic_hidden_depth=3 --actor_hidden_dim=256 --actor_hidden_depth=3 --num_interact=$num_interact --max_feedback="$feedback" --reward_batch="$reward_batch" --reward_update=10 --feed_type=1 --eps_mistake="$eps_mistake" --rho=$rho >> "./reward_model_log_RUNE_env_"$envname"_feedback_"$feedback"_mistake_$eps_mistake.txt" 2>&1
            ;;
        *)
            python train_RUNE.py --device=$device --env="$envname" --seed="$seed" --actor_lr=$sac_lr --critic_lr=$sac_lr --unsup_steps=9000 --steps=1000000 --num_interact=$num_interact --max_feedback="$feedback" --reward_batch="$reward_batch" --reward_update=10 --feed_type=1 --eps_mistake="$eps_mistake" --rho=$rho >> "./reward_model_log_RUNE_env_"$envname"_feedback_"$feedback"_mistake_$eps_mistake.txt" 2>&1
            ;;
        esac
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

seeds=(12345 23451 34512 45123 51234 67890 68790 78906 89067 90678)

for seed in "${seeds[@]}"; do
    run_command "$seed" &
done

wait
