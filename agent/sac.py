import numpy as np
import torch
import torch.nn.functional as F
from transformers import get_constant_schedule_with_warmup
import utils

from agent import Agent
from agent.critic import DoubleQCritic
from agent.actor import DiagGaussianActor

def compute_state_entropy(obs, full_obs, k):
    batch_size = 500
    with torch.no_grad():
        dists = []
        for idx in range(len(full_obs) // batch_size + 1):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            dist = torch.norm(
                obs[:, None, :] - full_obs[None, start:end, :], dim=-1, p=2
            )
            dists.append(dist)

        dists = torch.cat(dists, dim=1)
        knn_dists = torch.kthvalue(dists, k=k + 1, dim=1).values
        state_entropy = knn_dists
    return state_entropy.unsqueeze(1)


class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, cfg):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.device = torch.device(cfg.device)
        self.cfg = cfg
        self.discount = cfg.discount
        self.critic_tau = cfg.critic_tau
        self.actor_update_frequency = cfg.actor_update_frequency
        self.critic_target_update_frequency = cfg.critic_target_update_frequency
        self.batch_size = cfg.batch_size
        self.learnable_temperature = cfg.learnable_temperature
        self.critic_lr = cfg.critic_lr
        self.critic_betas = cfg.critic_betas
        self.s_ent_stats = torch.tensor([np.inf, -np.inf], device=cfg.device) # min, max
        self.state_ent = utils.TorchRunningMeanStd(shape=[1], device=cfg.device)
        self.init_temperature = cfg.init_temperature
        self.alpha_lr = cfg.alpha_lr
        self.alpha_betas = cfg.alpha_betas
        self.actor_betas = cfg.actor_betas
        self.critic_hidden_dim = cfg.critic_hidden_dim
        self.critic_hidden_depth = cfg.critic_hidden_depth
        self.actor_hidden_dim = cfg.actor_hidden_dim
        self.actor_hidden_depth = cfg.actor_hidden_depth
        self.actor_log_std_bounds = cfg.actor_log_std_bounds
        
        self.critic = DoubleQCritic(
            obs_dim, action_dim, cfg.critic_hidden_dim, cfg.critic_hidden_depth).to(
            self.device)
        self.critic_target = DoubleQCritic(
            obs_dim, action_dim, cfg.critic_hidden_dim, cfg.critic_hidden_depth).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = DiagGaussianActor(
            obs_dim, action_dim, cfg.actor_hidden_dim, cfg.actor_hidden_depth,
            cfg.actor_log_std_bounds).to(self.device)
        self.log_alpha = torch.tensor(np.log(cfg.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=cfg.actor_lr,
            betas=cfg.actor_betas)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=cfg.critic_lr,
            betas=cfg.critic_betas)
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=cfg.alpha_lr,
            betas=cfg.alpha_betas)
        
        # change mode
        self.train()
        self.critic_target.train()
    
    def reset_critic(self):
        self.critic = DoubleQCritic(
            self.obs_dim, self.action_dim, self.critic_hidden_dim, 
            self.critic_hidden_depth).to(self.device)
        self.critic_target = DoubleQCritic(
            self.obs_dim, self.action_dim, self.critic_hidden_dim, 
            self.critic_hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            betas=self.critic_betas)
    
    def reset_actor(self):
        # reset log_alpha
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=self.alpha_lr,
            betas=self.alpha_betas)
        
        # reset actor
        self.actor = DiagGaussianActor(
            self.obs_dim, self.action_dim, self.actor_hidden_dim, 
            self.actor_hidden_depth, self.actor_log_std_bounds).to(
            self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_lr,
            betas=self.actor_betas)
        
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, 
                      not_done, logger, step, print_flag=True):
        
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        
        if print_flag:
            logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.log(logger, step)
        
    def update_critic_state_ent(
        self, obs, full_obs, action, next_obs, not_done, logger,
        step, K=5, print_flag=True):
        
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        
        # compute state entropy
        state_entropy = compute_state_entropy(obs, full_obs, k=K)
        state_entropy_max = state_entropy.max()
        state_entropy_min = state_entropy.min()
        if print_flag:
            logger.log("train_critic/entropy", state_entropy.mean(), step)
            logger.log("train_critic/entropy_max", state_entropy_max, step)
            logger.log("train_critic/entropy_min", state_entropy_min, step)
        
        if state_entropy_max > self.s_ent_stats[1]:
            self.s_ent_stats[1] = state_entropy_max
        if state_entropy_min < self.s_ent_stats[0]:
            self.s_ent_stats[0] = state_entropy_min
        
        self.state_ent.update(state_entropy)

        # normalize to [-1, 1]
        norm_state_entropy = (state_entropy - self.state_ent.mean) / (3*self.state_ent.std)
        torch.clip(norm_state_entropy, -1, 1, out=norm_state_entropy)
        # scale = ((self.s_ent_stats - self.state_ent.mean) / self.state_ent.std).abs().max()
        # norm_state_entropy /= scale
        # norm_state_entropy = ((state_entropy - self.s_ent_stats[0]) / (self.s_ent_stats[1] - self.s_ent_stats[0]) - 0.5 ) * 2
        
        if print_flag:
            logger.log("train_critic/norm_entropy", norm_state_entropy.mean(), step)
            logger.log("train_critic/norm_entropy_max", norm_state_entropy.max(), step)
            logger.log("train_critic/norm_entropy_min", norm_state_entropy.min(), step)
                
        target_Q = norm_state_entropy + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        
        if print_flag:
            logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)
        return norm_state_entropy.cpu().detach().numpy()
    
    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic_target.state_dict(), '%s/critic_target_%s.pt' % (model_dir, step)
        )
        
    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.critic_target.load_state_dict(
            torch.load('%s/critic_target_%s.pt' % (model_dir, step))
        )
    
    def update_actor_and_alpha(self, obs, logger, step, print_flag=False):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        if print_flag:
            logger.log('train_actor/loss', actor_loss, step)
            logger.log('train_actor/target_entropy', self.target_entropy, step)
            logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            if print_flag:
                logger.log('train_alpha/loss', alpha_loss, step)
                logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            
    def update(self, replay_buffer, logger, step, gradient_update=1):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                               logger, step, print_flag)

            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
            
    def update_after_reset(self, replay_buffer, logger, step, gradient_update=1, policy_update=True):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                               logger, step, print_flag)

            if index % self.actor_update_frequency == 0 and policy_update:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

            if index % self.critic_target_update_frequency == 0:
                utils.soft_update_params(self.critic, self.critic_target,
                                         self.critic_tau)
            
    def update_state_ent(self, replay_buffer, logger, step, gradient_update=1, K=5):
        for index in range(gradient_update):
            obs, full_obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample_state_ent(
                self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            state_entropy = self.update_critic_state_ent(
                obs, full_obs, action, next_obs, not_done_no_max,
                logger, step, K=K, print_flag=print_flag)

            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
        return obs.cpu().detach().numpy(), action.cpu().detach().numpy(), state_entropy