from torch.cuda import is_available
from time import time

class RUNEConfig:
    def __init__(self, args):
        # variable parameters
        self.seed = args.seed or 42
        self.device = args.device or ('cuda' if is_available() else 'cpu')
        self.teacher_eps_equal = args.eps_equal or 0.0
        self.teacher_eps_mistake = args.eps_mistake or 0.0
        self.env = args.env or 'walker_walk'
        self.actor_lr = args.actor_lr or 0.0005
        self.critic_lr = args.critic_lr or 0.0005
        self.num_unsup_steps = args.unsup_steps or 9000
        self.num_train_steps = args.steps or 500000
        self.num_interact = args.num_interact or 20000  # frequency of teacher feedback
        self.max_feedback = args.max_feedback or 1000
        self.reward_batch = args.reward_batch or 100
        self.reward_update = args.reward_update or 50
        self.batch_size = args.batch_size or 1024  # 1024 for Walker, 512 for Meta-world
        self.critic_hidden_dim = args.critic_hidden_dim or 1024
        self.critic_hidden_depth = args.critic_hidden_depth or 2
        self.actor_hidden_dim = args.actor_hidden_dim or 1024
        self.actor_hidden_depth = args.actor_hidden_depth or 2
        self.feed_type = args.feed_type or 0  # 0: random, 1: uncertainty, 2: entropy
        self.ensemble_size = args.ensemble_size or 3
        self.beta_decay = args.rho or 0.00001
        
        self.beta_init = 0.05
        self.beta_schedule = 'linear_decay'

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
        self.num_seed_steps = 1000

        # eval settings
        self.eval_frequency = 10000
        self.num_eval_episodes = 10

        # reward model
        self.segment = 50
        self.activation = 'tanh'
        self.reward_lr = 0.0003
        self.large_batch = 10
        self.label_margin = 0.0
        self.teacher_beta = -1
        self.teacher_gamma = 1
        self.teacher_eps_skip = 0
        self.reset_update = 100

        # log settings
        self.log_save_tb = False
        self.log_frequency = 1000
        self.eval_log_name = f'eval_RUNE_mistake_{self.teacher_eps_mistake}_seed_{self.seed}_{time()}'
        self.train_log_name = f'train_RUNE_mistake_{self.teacher_eps_mistake}_seed_{self.seed}_{time()}'

        # scheduling
        self.reward_schedule = 0

        # video recorder
        self.save_video = False

        # Environment
        self.gradient_update=1
