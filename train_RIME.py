import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import utils
from agent.sac_RIME import SACAgent, compute_state_entropy
from config.RIME import RIMEConfig
from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model_RIME import RIMERewardModel, set_device_RIME


class Workspace:
    def __init__(self, cfg):
        self.work_dir = os.path.join(os.getcwd(), 'RIME', cfg.env)
        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent_name,
            train_log_name=cfg.train_log_name,
            eval_log_name=cfg.eval_log_name,
        )

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg.env, cfg.seed)
            self.log_success = True
            k = 600
            tau = 0.001
        else:
            self.env = utils.make_env(cfg.env, cfg.seed)
            self.log_success = False
            k = 60
            tau = 0.001
        
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = SACAgent(
            obs_dim, action_dim, action_range, cfg
        )

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device
        )
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiating the reward model
        self.reward_model = RIMERewardModel(
            cfg.seed,
            device=self.device,
            k=k,
            threshold_variance=cfg.threshold_variance,
            threshold_alpha=cfg.threshold_alpha,
            threshold_beta_init=cfg.threshold_beta_init,
            threshold_beta_min=cfg.threshold_beta_min,
            flipping_tau=tau,
            num_warmup_steps=int(1/3*cfg.max_feedback/cfg.reward_batch*cfg.least_reward_update+0.5),
            ds=self.env.observation_space.shape[0],
            da=self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal,
        )
        
    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        num_eval_episodes = self.cfg.num_eval_episodes
        
        for episode in range(num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))
                obs, reward, done, extra = self.env.step(action)
                
                episode_reward += reward_hat
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= num_eval_episodes
        average_true_episode_reward /= num_eval_episodes
        if self.log_success:
            success_rate /= num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        self.logger.log('eval/num_eval_episodes', num_eval_episodes,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                            self.step)
            self.logger.log('train/true_episode_success', success_rate,
                            self.step)
        self.logger.dump(self.step)
    
    def learn_reward(self, first_flag=0):
                
        # get feedbacks
        labeled_queries = 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            elif self.cfg.feed_type == 6:
                labeled_queries = self.reward_model.uniform_sampling()
                raise NotImplementedError
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                debug = False
                if epoch % 5 == 0 or epoch == self.cfg.reward_update - 1:
                    debug = True
                
                train_acc = self.reward_model.train_reward(debug=debug)
                total_acc = np.mean(train_acc)
                
                # early stop
                if total_acc > 0.98 and epoch > self.cfg.least_reward_update:
                    break
                    
        print(f"Reward function is updated!! ACC: {total_acc:.4f}, Epoch: {epoch}")

    def run(self):
        episode, episode_reward, done, intrinsic_reward = 0, 0, True, None
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 
        start_time = time.time()

        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            
            # if done, log & evaluate & reset
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                self.logger.log('train/mistake_feedback', self.reward_model.mistake_labels, self.step)
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                                    self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                                    self.step)
                
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
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
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)

                # relabel buffer due to training of reward model during unsup steps
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                
                # first learn reward
                self.reward_model.set_lr_schedule()
                self.learn_reward(first_flag=1)
                
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0
            
            # 3 differences from above: first_flag, corner case, update method (reset critic)
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0

                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(
                    self.replay_buffer, self.logger, self.step,
                    gradient_update=1, K=self.cfg.topK)
                
                # update reward model to fit with intrinsic reward
                for _ in range(5):
                    unsup_obs, full_obs, unsup_act, _, _, _, _ = self.replay_buffer.sample_state_ent(
                        self.agent.batch_size)
                    
                    state_entropy = compute_state_entropy(unsup_obs, full_obs, k=self.cfg.topK)
                    norm_state_entropy = (state_entropy - self.agent.state_ent.mean) / self.agent.state_ent.std
                    scale = ((self.agent.s_ent_stats - self.agent.state_ent.mean) / self.agent.state_ent.std).abs().max()
                    norm_state_entropy /= scale
                    
                    self.reward_model.opt.zero_grad()
                    unsup_rew_loss = 0.0
                    for member in range(self.reward_model.de):
                        rew_hat = self.reward_model.ensemble[member](torch.cat([unsup_obs, unsup_act], dim=-1).to(self.device))
                        unsup_rew_loss += F.mse_loss(rew_hat, norm_state_entropy.detach().to(self.device))
                    unsup_rew_loss.backward()
                    self.reward_model.opt.step()
                
            next_obs, reward, done, extra = self.env.step(action)
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1
        
        save_dir = os.path.join(self.work_dir, f'mistake_{self.cfg.teacher_eps_mistake}_seed_{self.cfg.seed}')
        os.makedirs(save_dir, exist_ok=True)
        self.agent.save(save_dir, self.step)
        self.reward_model.save(save_dir, self.step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--least_reward_update', type=int)
    parser.add_argument('--threshold_variance', type=str)
    parser.add_argument('--threshold_alpha', type=float)
    parser.add_argument('--threshold_beta_init', type=float)
    parser.add_argument('--threshold_beta_min', type=float)
    
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', type=str)
    parser.add_argument('--env', type=str)
    parser.add_argument('--eps_mistake', type=float)
    parser.add_argument('--eps_equal', type=float)
    parser.add_argument('--eps_skip', type=float)
    parser.add_argument('--teacher_gamma', type=float)
    parser.add_argument('--actor_lr', type=float)
    parser.add_argument('--critic_lr', type=float)
    parser.add_argument('--unsup_steps', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--num_interact', type=int)
    parser.add_argument('--max_feedback', type=int)
    parser.add_argument('--reward_batch', type=int)
    parser.add_argument('--reward_update', type=int)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('--critic_hidden_dim', type=int)
    parser.add_argument('--actor_hidden_dim', type=int)
    parser.add_argument('--critic_hidden_depth', type=int)
    parser.add_argument('--actor_hidden_depth', type=int)
    parser.add_argument('--feed_type', type=int)
    parser.add_argument('--ensemble_size', type=int)
    args = parser.parse_args()
    cfg = RIMEConfig(args)
    set_device_RIME(cfg.device)
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
