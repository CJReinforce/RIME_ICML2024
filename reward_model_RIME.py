import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from reward_model import RewardModel, device, set_device
from transformers import get_constant_schedule_with_warmup
from utils import RunningMeanStd


def set_device_RIME(dev):
    global device
    device = dev
    set_device(dev)


class RIMERewardModel(RewardModel):
    def __init__(
            self, 
            seed,
            k,
            device='cuda' if torch.cuda.is_available() else 'cpu', 
            threshold_variance='kl', 
            threshold_alpha=0.5,
            threshold_beta_init=3.0,
            threshold_beta_min=1.0,
            flipping_tau=0.001,
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

        self.update_step = 0

        # debug
        self.history_info = {}
    
    def set_lr_schedule(self):
        self.lr_schedule = get_constant_schedule_with_warmup(self.opt, self.num_warmup_steps)

    def get_threshold_beta(self):
        return max(self.threshold_beta_min, -(self.threshold_beta_init-self.threshold_beta_min)/self.k * self.update_step + self.threshold_beta_init)
    
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
                p_hat_all.append(F.softmax(r_hat, dim=-1).cpu())
        
        # predict label for all ensemble members
        p_hat_all = torch.stack(p_hat_all)  # (de, max_len, 2)
        
        # compute KL divergence
        predict_label = p_hat_all.mean(0)  # (max_len, 2)
        if self.label_margin > 0 or self.teacher_eps_equal > 0:
            buffer_label = torch.tensor(self.buffer_label[:max_len].flatten()).long()
            target_label = torch.zeros_like(predict_label)
            temp_buffer_label = torch.clamp(buffer_label, min=0)
            target_label.scatter_(1, temp_buffer_label.unsqueeze(1), 1)
            mask = buffer_label == -1
            target_label[mask, :] = 0.5
        else:
            target_label = torch.zeros_like(predict_label).scatter(1, torch.from_numpy(self.buffer_label[:max_len].flatten()).long().unsqueeze(1), 1)
        
        KL_div = (-target_label * torch.log(predict_label)).sum(1)  # (max_len,)
        
        # filter trust samples
        x = self.KL_div.max
        baseline = -np.log(x + 1e-8) + self.threshold_alpha * x
        if self.threshold_variance == 'prob':
            uncertainty = self.get_threshold_beta() * predict_label[:, 0].std(0)
        else:
            uncertainty = min(self.get_threshold_beta() * self.KL_div.var, 3.0)
        trust_sample_bool_index = KL_div < baseline + uncertainty
        trust_sample_index = np.where(trust_sample_bool_index)[0]

        # label flipping
        flipping_threshold = -np.log(self.flipping_tau)
        flipping_sample_bool_index = KL_div > flipping_threshold
        flipping_sample_index = np.where(flipping_sample_bool_index)[0]
        
        # update KL divergence statistics of trust samples
        self.KL_div.update(KL_div[trust_sample_bool_index].numpy())

        ## debug info start...
        # accuracy of predicted trust samples
        accurate_samples = (self.buffer_label[:max_len][trust_sample_bool_index] == self.buffer_GT_label[:max_len][trust_sample_bool_index]).sum()
        trust_samples = len(trust_sample_index)
        accuracy = accurate_samples / trust_samples
        # recall of predicted trust samples
        recall_samples = (self.buffer_label[:max_len][~trust_sample_bool_index] == self.buffer_GT_label[:max_len][~trust_sample_bool_index]).sum()
        non_trust_samples = max_len - trust_samples
        recall = recall_samples / non_trust_samples if non_trust_samples > 0 else 0.0
        # flipping accuracy
        flipping_correct = (1 - self.buffer_label[flipping_sample_index] == self.buffer_GT_label[flipping_sample_index]).sum()
        flipping_samples = len(flipping_sample_index)
        flipping_accuracy = flipping_correct / flipping_samples if flipping_samples > 0 else 0.0

        if debug:
            print('#'*10, self.seed, '#'*10)
            self.history_info = {
                'trust sample ratio': len(trust_sample_index)/max_len, 
                'KL_div': self.KL_div.max, 
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
        ## debug info end...

        if trust_sample and label_flipping:
            # temporarily flipping
            self.buffer_label[flipping_sample_index] = 1-self.buffer_label[flipping_sample_index]
            training_sample_index = np.concatenate([trust_sample_index, flipping_sample_index])
        elif not trust_sample and label_flipping:
            # temporarily flipping
            self.buffer_label[flipping_sample_index] = 1-self.buffer_label[flipping_sample_index]
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
                if self.label_margin > 0 or self.teacher_eps_equal > 0:
                    uniform_index = labels == -1
                    labels[uniform_index] = 0
                    target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                    target_onehot += self.label_margin
                    if uniform_index.int().sum().item() > 0:
                        target_onehot[uniform_index] = 0.5
                    curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                else:
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
        if label_flipping:
            self.buffer_label[flipping_sample_index] = 1-self.buffer_label[flipping_sample_index]
        
        ensemble_acc = ensemble_acc / total
        self.update_step += 1
        
        return ensemble_acc