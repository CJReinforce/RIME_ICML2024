import argparse
import os
import time
from collections import deque

import numpy as np
import gym
import torch
import torch.nn.functional as F
from transformers import get_constant_schedule_with_warmup

import utils
from agent.sac import SACAgent, compute_state_entropy
from config.RIME import RIMEConfig
from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel, set_device


def get_upper_quarter(data):
    return np.percentile(data, 100)

class RunningMeanStd:
    def __init__(self, mean=0, std=1.0, epsilon=np.finfo(np.float32).eps.item(), 
                 mode='common', lr=0.1):
        self.mean, self.var = mean, std
        self.max, self.quarter = mean, mean
        self.count = 0
        self.eps = epsilon
        self.lr = lr
        self.mode = mode

    def print_info(self):
        return {'mean': self.mean, 'var': self.var, 'max': self.max, 'quarter': self.quarter}

    def update(self, data_array) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = np.mean(data_array, axis=0), np.var(data_array, axis=0)
        batch_count = len(data_array)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        if self.mode == 'common':
            new_mean = self.mean + delta * batch_count / total_count
            new_max = self.max + (np.max(data_array)-self.max) / total_count
            new_quarter = self.quarter + (get_upper_quarter(data_array)-self.quarter) / total_count
        else:
            new_mean = self.mean + delta * self.lr
            new_max = self.max + (np.max(data_array)-self.max) * self.lr
            new_quarter = self.quarter + (get_upper_quarter(data_array)-self.quarter) * self.lr
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count
        self.max = new_max
        self.quarter = new_quarter


class OurRewardModel(RewardModel):
    def __init__(
            self, 
            seed:int,
            k=600,
            device='cuda' if torch.cuda.is_available() else 'cpu', 
            threshold_variance='prob', 
            threshold_alpha=0.5,
            threshold_beta_init=1.0,
            threshold_beta_min=0.1,
            flipping_tau=0.01,
            num_warmup_steps=50,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.lr_schedule = None
        self.seed = seed
        self.device = device
        self.k = k
        self.KL_div = RunningMeanStd(mode='fixed', lr=0.1)
        assert threshold_variance.lower() in ['kl', 'prob']
        self.threshold_variance = threshold_variance
        self.threshold_alpha = threshold_alpha
        self.threshold_beta_init = threshold_beta_init
        self.threshold_beta_min = threshold_beta_min
        self.flipping_tau = flipping_tau
        self.num_warmup_steps = num_warmup_steps
        
        config = {
            'device': self.device, 
            'threshold_variance': self.threshold_variance, 
            'threshold_alpha': self.threshold_alpha, 
            'threshold_beta_init': self.threshold_beta_init, 
            'threshold_beta_min': self.threshold_beta_min, 
            'eps_mistake': self.teacher_eps_mistake, 
            'eps_equal': self.teacher_eps_equal, 
            # 'lr_schedule': self.lr_schedule.get_last_lr()[0],
            'warmup_steps': num_warmup_steps,
            'seed': self.seed
        }
        print(f'Reward model config: {config}')
        self.update_step = 0
        self.trust_sample_index = None

        # debug
        self.history_info = {}
    
    def set_lr_schedule(self):
        self.lr_schedule = get_constant_schedule_with_warmup(self.opt, self.num_warmup_steps)

    def get_threshold_beta(self):
        return max(self.threshold_beta_min, -(self.threshold_beta_init-self.threshold_beta_min)/self.k*self.update_step + self.threshold_beta_init)
    
    def train_reward(self, debug=False, trust_sample=True, label_flipping=True):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index

        # compute trust samples
        p_hat_all = []
        with torch.no_grad():
            for member in range(self.de):
                r_hat1 = self.r_hat_member(self.buffer_seg1[:max_len], member=member)
                r_hat2 = self.r_hat_member(self.buffer_seg2[:max_len], member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)  # (max_len, 2)
                assert r_hat.shape == (max_len, 2)
                p_hat_all.append(F.softmax(r_hat, dim=-1).cpu())
        
        # predict label for all ensemble members
        p_hat_all = torch.stack(p_hat_all)  # (de, max_len, 2)
        assert p_hat_all.shape == (self.de, max_len, 2)
        
        # compute KL divergence
        predict_label = p_hat_all.mean(0)  # (max_len, 2)
        assert predict_label.shape == (max_len, 2)
        target_label = torch.zeros_like(predict_label).scatter(1, torch.from_numpy(self.buffer_label[:max_len].flatten()).long().unsqueeze(1), 1)
        KL_div = (-target_label * torch.log(predict_label)).sum(1)  # (max_len,)
        
        # filter trust samples
        x = self.KL_div.quarter
        # baseline = 1 - self.threshold_alpha*x
        baseline = -np.log(x+1e-8) + self.threshold_alpha*x
        if self.threshold_variance == 'prob':
            uncertainty = self.get_threshold_beta() * predict_label[:, 0].std(0)
        else:
            uncertainty = min(self.get_threshold_beta() * self.KL_div.var, 3.0)
        trust_sample_bool_index = KL_div < baseline + uncertainty
        trust_sample_index = np.where(trust_sample_bool_index)[0]
        self.trust_sample_index = trust_sample_index

        # label flipping
        # x = self.KL_div.max
        # baseline_flipping = max(1.5, -np.log(x) + x/2, baseline)
        flipping_threshold = -np.log(self.flipping_tau)  # baseline_flipping + uncertainty
        flipping_sample_bool_index = KL_div > flipping_threshold
        flipping_sample_index = np.where(flipping_sample_bool_index)[0]
        
        # update KL divergence statistics of trust samples
        self.KL_div.update(KL_div[trust_sample_bool_index].numpy())

        # accuracy of predicted trust samples
        accurate_samples = (self.buffer_label[:max_len][trust_sample_bool_index] == self.buffer_GT_label[:max_len][trust_sample_bool_index]).sum()
        trust_samples = len(trust_sample_index)
        accuracy = accurate_samples / trust_samples
        # recall of predicted trust samples
        recall_samples = (self.buffer_label[:max_len][~trust_sample_bool_index] == self.buffer_GT_label[:max_len][~trust_sample_bool_index]).sum()
        non_trust_samples = max_len - trust_samples
        recall = recall_samples / non_trust_samples if non_trust_samples > 0 else 0.0
        # flipping accuracy
        # temporarily flipping
        self.buffer_label[flipping_sample_index] = 1-self.buffer_label[flipping_sample_index]
        flipping_correct = (self.buffer_label[flipping_sample_index] == self.buffer_GT_label[flipping_sample_index]).sum()
        flipping_samples = len(flipping_sample_index)
        flipping_accuracy = flipping_correct / flipping_samples if flipping_samples > 0 else 0.0

        if debug:
            print('#'*10, self.seed, '#'*10)
            self.history_info = {
                'trust sample ratio': len(trust_sample_index)/max_len, 
                'KL_div': self.KL_div.print_info(), 
                'predict_label_div': predict_label[:, 0].std(0).item(), 
                'beta': self.get_threshold_beta(), 
                'baseline': baseline, 
                'uncertainty': uncertainty, 
                'threshold': baseline+uncertainty, 
                'flipping_threshold': flipping_threshold, 
                'update_step': self.update_step,
                'lr': self.lr_schedule.get_last_lr()[0],
                'seed': self.seed,
                'accuracy': [accurate_samples, trust_samples, accuracy], 
                'recall': [recall_samples, non_trust_samples, recall], 
                'flipping_accuracy': [flipping_correct, flipping_samples, flipping_accuracy]
            }
            print(self.history_info)
        
        if trust_sample and label_flipping:
            training_sample_index = np.concatenate([trust_sample_index, flipping_sample_index])
        elif not trust_sample and label_flipping:
            training_sample_index = np.arange(max_len)
        elif trust_sample and not label_flipping:
            training_sample_index = trust_sample_index
        else:
            training_sample_index = np.arange(max_len)
        
        max_len = len(training_sample_index)
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(training_sample_index))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        self.lr_schedule.step()
        
        # change back
        self.buffer_label[flipping_sample_index] = 1-self.buffer_label[flipping_sample_index]
        
        ensemble_acc = ensemble_acc / total
        self.update_step += 1
        
        return ensemble_acc


def reward_fn(a, ob):
    backroll = -ob[7]
    height = ob[0]
    vel_act = a[0] * ob[8] + a[1] * ob[9] + a[2] * ob[10]
    backslide = -ob[5]
    return backroll * (1.0 + .3 * height + .1 * vel_act + .05 * backslide)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        self.cfg = cfg
        self.logger = Logger(
            os.path.join(self.work_dir, 'RIME_human', cfg.env),
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent_name,
            train_log_name=cfg.train_log_name,
            eval_log_name=cfg.eval_log_name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        
        # make env
        self.env = gym.make('Hopper-v3', 
                            exclude_current_positions_from_observation=False,
                            terminate_when_unhealthy=False)
        env_record = gym.make('Hopper-v3', 
                              exclude_current_positions_from_observation=False,
                              terminate_when_unhealthy=False)
        self.env.seed(cfg.seed)
        env_record.seed(cfg.seed)
        self.log_success = False
        k = 30
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
            self.device)
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiating the reward model
        self.reward_model = OurRewardModel(
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
            env=env_record,
            video_recoder_dir='cj',
        )
        
    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        num_eval_episodes = self.cfg.num_eval_episodes if self.step < self.cfg.num_train_steps - 10*self.cfg.eval_frequency else 100
        
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
                next_obs, _, done, extra = self.env.step(action)
                reward = reward_fn(action, obs[1:])
                obs = next_obs
                
                episode_reward += reward
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
        if first_flag == 1:
            labeled_queries = self.reward_model.uniform_sampling_with_human_labeller()
        else:
            labeled_queries = self.reward_model.disagreement_sampling_with_human_labeller()
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        if self.labeled_feedback > 0:
            
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
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
        episode, episode_reward, done = 0, 0, True
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
                
                # save the last trust trajectories
                # elif not self.saved_tau:
                #     seg1 = self.reward_model.buffer_seg1.copy()
                #     seg2 = self.reward_model.buffer_seg2.copy()
                #     trust_seg1 = seg1[self.reward_model.trust_sample_index].reshape(len(self.reward_model.trust_sample_index), -1)
                #     trust_seg2 = seg2[self.reward_model.trust_sample_index].reshape(len(self.reward_model.trust_sample_index), -1)
                #     trust_seg = np.vstack((trust_seg1, trust_seg2))
                #     dir_name = os.path.join(self.work_dir, 'saved_trajectories', 'RIME_lr_schedule', f'mistake_{self.cfg.teacher_eps_mistake}')
                #     os.makedirs(dir_name, exist_ok=True)
                #     random = time.time()
                #     np.save(os.path.join(dir_name, f'trust_tau_seed_{self.cfg.seed}_{random}.npy'), trust_seg)
                #     self.saved_tau = True

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
                
                if self.step % 500 == 0:
                    print(f"state entropy loss: {unsup_rew_loss.item()}, step: {self.step}, seed: {self.cfg.seed}")
                
            next_obs, _, done, extra = self.env.step(action)
            reward = reward_fn(action, obs[1:])
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

            # debug
            if self.step % 100_000 == 0:
                print('Debug:', self.reward_model.history_info)
        
        agent_save_dir = os.path.join(self.work_dir, 'RIME_human', self.cfg.env, 'agent')
        reward_save_dir = os.path.join(self.work_dir, 'RIME_human', self.cfg.env, 'reward')
        os.makedirs(agent_save_dir, exist_ok=True)
        os.makedirs(reward_save_dir, exist_ok=True)
        self.agent.save(agent_save_dir, self.step)
        self.reward_model.save(reward_save_dir, self.step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', type=str)
    parser.add_argument('--env', type=str)
    parser.add_argument('--least_reward_update', type=int)
    parser.add_argument('--threshold_variance', type=str)
    parser.add_argument('--threshold_alpha', type=float)
    parser.add_argument('--threshold_beta_init', type=float)
    parser.add_argument('--threshold_beta_min', type=float)
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
    set_device(cfg.device)
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
