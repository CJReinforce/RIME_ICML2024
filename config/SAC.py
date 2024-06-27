from torch.cuda import is_available


class SACConfig:
    def __init__(self, args):
        # variable parameters
        self.batch_size = args.batch_size or 1024  # 1024 for Walker, 512 for Meta-world
        self.num_train_steps = args.steps or 500000
        self.critic_hidden_dim = args.critic_hidden_dim or 1024
        self.critic_hidden_depth = args.critic_hidden_depth or 2
        self.actor_hidden_depth = args.actor_hidden_depth or 2
        self.actor_hidden_dim = args.actor_hidden_dim or 1024
        self.env = args.env or 'walker_walk'
        self.seed = args.seed or 42
        self.device = args.device or ('cuda' if is_available() else 'cpu')
        self.actor_lr = args.actor_lr or 0.0005
        self.critic_lr = args.critic_lr or 0.0005

        # agent settings
        self.agent_name = 'sac'
        self.discount = 0.99
        self.init_temperature = 0.1
        self.alpha_lr = 1e-4
        self.alpha_betas = [0.9, 0.999]
        self.actor_betas = [0.9, 0.999]
        self.actor_update_frequency = 1
        self.critic_betas = [0.9, 0.999]
        self.critic_tau = 0.005
        self.critic_target_update_frequency = 2
        self.learnable_temperature = True
        self.actor_log_std_bounds = [-5, 2]
        self.topK = 5

        # steps settings
        self.replay_buffer_capacity = self.num_train_steps
        self.num_seed_steps = 5000
        self.num_unsup_steps = 0

        # eval settings
        self.eval_frequency = 10000
        self.num_eval_episodes = 10
        self.reset_update = 100

        # log settings
        self.log_save_tb = False
        self.log_frequency = 1000
        self.eval_log_name = f'eval_SAC_seed_{self.seed}'
        self.train_log_name = f'train_SAC_seed_{self.seed}'

        # video recorder
        self.save_video = False
        self.save_model = True
