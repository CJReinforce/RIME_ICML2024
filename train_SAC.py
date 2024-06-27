import argparse
import os
import time

import numpy as np
import torch
import utils
from agent.sac import SACAgent
from config.SAC import SACConfig
from logger import Logger
from replay_buffer import ReplayBuffer


class Workspace:
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        self.cfg = cfg
        self.logger = Logger(
            os.path.join(self.work_dir, 'SAC', cfg.env),
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent_name,
            train_log_name=cfg.train_log_name,
            eval_log_name=cfg.eval_log_name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.step = 0
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg.env, cfg.seed)
            self.log_success = True
        else:
            self.log_success = False
            self.env = utils.make_env(cfg.env, cfg.seed)

        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = SACAgent(
            obs_dim, action_dim, action_range, cfg
        )
        
        # no relabel
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)

    def evaluate(self):
        average_episode_reward = 0
        if self.log_success:
            success_rate = 0
        num_eval_episodes = self.cfg.num_eval_episodes
        
        for episode in range(num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)
                episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

            average_episode_reward += episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= num_eval_episodes
        if self.log_success:
            success_rate /= num_eval_episodes
            success_rate *= 100.0

        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/num_eval_episodes', num_eval_episodes,
                        self.step)
        
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                        self.step)
        self.logger.dump(self.step)
        
    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        start_time = time.time()
        fixed_start_time = time.time()
        
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    self.logger.log('train/total_duration',
                                    time.time() - fixed_start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                            
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update             
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps) and self.cfg.num_unsup_steps > 0:
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
            
            
            next_obs, reward, done, extra = self.env.step(action)      
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            self.replay_buffer.add(
                obs, action, 
                reward, next_obs, done,
                done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--critic_hidden_dim', type=int)
    parser.add_argument('--actor_hidden_dim', type=int)
    parser.add_argument('--critic_hidden_depth', type=int)
    parser.add_argument('--actor_hidden_depth', type=int)
    parser.add_argument('--env', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', type=str)
    parser.add_argument('--actor_lr', type=float)
    parser.add_argument('--critic_lr', type=float)
    args = parser.parse_args()
    cfg = SACConfig(args)
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
