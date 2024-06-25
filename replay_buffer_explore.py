import numpy as np
import torch
import utils

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, window=1):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.extrinsic_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.intrinsic_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.window = window

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, ext_reward, int_reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.extrinsic_rewards[self.idx], ext_reward)
        np.copyto(self.intrinsic_rewards[self.idx], int_reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def add_batch(self, obs, action, ext_reward, int_reward, next_obs, done, done_no_max):
        
        next_index = self.idx + self.window
        if next_index >= self.capacity:
            self.full = True
            maximum_index = self.capacity - self.idx
            np.copyto(self.obses[self.idx:self.capacity], obs[:maximum_index])
            np.copyto(self.actions[self.idx:self.capacity], action[:maximum_index])
            np.copyto(self.extrinsic_rewards[self.idx:self.capacity], ext_reward[:maximum_index])
            np.copyto(self.intrinsic_rewards[self.idx:self.capacity], int_reward[:maximum_index])
            np.copyto(self.next_obses[self.idx:self.capacity], next_obs[:maximum_index])
            np.copyto(self.not_dones[self.idx:self.capacity], done[:maximum_index] <= 0)
            np.copyto(self.not_dones_no_max[self.idx:self.capacity], done_no_max[:maximum_index] <= 0)
            remain = self.window - (maximum_index)
            if remain > 0:
                np.copyto(self.obses[0:remain], obs[maximum_index:])
                np.copyto(self.actions[0:remain], action[maximum_index:])
                np.copyto(self.extrinsic_rewards[0:remain], ext_reward[maximum_index:])
                np.copyto(self.intrinsic_rewards[0:remain], int_reward[maximum_index:])
                np.copyto(self.next_obses[0:remain], next_obs[maximum_index:])
                np.copyto(self.not_dones[0:remain], done[maximum_index:] <= 0)
                np.copyto(self.not_dones_no_max[0:remain], done_no_max[maximum_index:] <= 0)
            self.idx = remain
        else:
            np.copyto(self.obses[self.idx:next_index], obs)
            np.copyto(self.actions[self.idx:next_index], action)
            np.copyto(self.extrinsic_rewards[self.idx:next_index], ext_reward)
            np.copyto(self.intrinsic_rewards[self.idx:next_index], int_reward)
            np.copyto(self.next_obses[self.idx:next_index], next_obs)
            np.copyto(self.not_dones[self.idx:next_index], done <= 0)
            np.copyto(self.not_dones_no_max[self.idx:next_index], done_no_max <= 0)
            self.idx = next_index
        
    def relabel_with_predictor(self, predictor):
        batch_size = 200
        total_iter = int(self.idx/batch_size)
        
        if self.idx > batch_size*total_iter:
            total_iter += 1
            
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > self.idx:
                last_index = self.idx
                
            obses = self.obses[index*batch_size:last_index]
            actions = self.actions[index*batch_size:last_index]
            inputs = np.concatenate([obses, actions], axis=-1)
            
            pred_reward = predictor.r_hat_batch(inputs)
            pred_reward, explore_bonus = predictor.r_hat_std_batch(inputs)
            self.extrinsic_rewards[index*batch_size:last_index] = pred_reward
            self.intrinsic_rewards[index*batch_size:last_index] = explore_bonus
            
    def relabel_with_bayes_predictor(self, predictor):
        batch_size = 200
        total_iter = int(self.idx/batch_size)
        
        if self.idx > batch_size*total_iter:
            total_iter += 1
            
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > self.idx:
                last_index = self.idx
                
            obses = self.obses[index*batch_size:last_index]
            actions = self.actions[index*batch_size:last_index]
            inputs = np.concatenate([obses, actions], axis=-1)
            
            pred_reward = predictor.r_hat(inputs)
            self.rewards[index*batch_size:last_index] = pred_reward

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        ext_rewards = torch.as_tensor(self.extrinsic_rewards[idxs], device=self.device)
        int_rewards = torch.as_tensor(self.intrinsic_rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, ext_rewards, int_rewards, next_obses, not_dones, not_dones_no_max
    

    def sample_combine(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        ext_rewards = torch.as_tensor(self.extrinsic_rewards[idxs], device=self.device)
        int_rewards = torch.as_tensor(self.intrinsic_rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        
        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[: self.idx]
        full_idxs = np.random.choice(full_obs.shape[0], size=512, replace=False)
        full_obs = torch.as_tensor(full_obs[full_idxs], device=self.device)

        return obses, full_obs, actions, ext_rewards, int_rewards, next_obses, not_dones, not_dones_no_max
 
    
    def sample_state_ent(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        ext_rewards = torch.as_tensor(self.extrinsic_rewards[idxs], device=self.device)
        int_rewards = torch.as_tensor(self.intrinsic_rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        
        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[: self.idx]
        full_obs = torch.as_tensor(full_obs, device=self.device)
        
        return obses, full_obs, actions, ext_rewards, int_rewards, next_obses, not_dones, not_dones_no_max
    
    def sample_full_obs(self):
        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[: self.idx]
        full_idxs = np.random.choice(full_obs.shape[0], size=512, replace=False)
        full_obs = torch.as_tensor(full_obs[full_idxs], device=self.device)
        return full_obs
    
    # extract future k timesteps of index from replay buffer, not including current index
    # return length of future timesteps actually
    def get_future_intrinsic_reward(self, index, k):
        # episode length of metaworld enviornment is 500
        # episode length of dm control enviornment is 1000
        remain = index % 500
        if 500 - remain - 1 >= k:
            idxs = range(index + 1, index + k + 1)
            assert len(idxs) == k
        else:
            idxs = range(index + 1, 500)

        # obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        # actions = torch.as_tensor(self.actions[idxs], device=self.device)
        # ext_rewards = torch.as_tensor(self.extrinsic_rewards[idxs], device=self.device)
        int_rewards = torch.as_tensor(self.intrinsic_rewards[idxs], device=self.device)
        # next_obses = torch.as_tensor(self.next_obses[idxs],
        #                              device=self.device).float()
        # not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        # not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
        #                                    device=self.device)

        # return obses, actions, ext_rewards, int_rewards, next_obses, not_dones, not_dones_no_max, len(idxs)
        return int_rewards, len(idxs)

class COACHReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, window, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, window, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, window, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, window, 1), dtype=np.float32)
        self.probs = np.empty((capacity, window, *action_shape), dtype=np.float32)
        self.window = window

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, prob):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.probs[self.idx], prob)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def sample(self, batch_size):
        idxs = np.random.randint(
            0,
            self.capacity if self.full else self.idx,
            size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        probs = torch.as_tensor(self.probs[idxs], device=self.device)

        return obses, actions, rewards, probs
    
class PixelReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        capacity,
        image_size,
        aug_type,
        device,
    ):
        self.capacity = capacity
        self.aug_type = aug_type
        self.image_size = image_size
        self.device = device

        self.candidates = dict()
        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, magnitude=None):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        raw_obses = self.obses[idxs]
        raw_next_obses = self.next_obses[idxs]
        
        if self.aug_type == "crop":
            raw_obses = utils.fast_random_crop(raw_obses, self.image_size)
            raw_next_obses = utils.fast_random_crop(raw_next_obses, self.image_size)
        else:
            obses = raw_obses
            next_obses = raw_next_obses
        
        obses = torch.as_tensor(raw_obses, device=self.device).float()
        next_obses = torch.as_tensor(raw_next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(
            self.not_dones_no_max[idxs], device=self.device
        )

        return (
            obses,
            actions,
            rewards,
            next_obses,
            not_dones_no_max,
        )
    
    def relabel_with_predictor(self, predictor):
        batch_size = 100
        total_iter = int(self.idx/batch_size)
        
        if self.idx > batch_size*total_iter:
            total_iter += 1
            
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > self.idx:
                last_index = self.idx
                
            obses = self.obses[index*batch_size:last_index]
            actions = self.actions[index*batch_size:last_index]
            print(obses.shape)
            print(actions.shape)
            pred_reward = predictor.r_hat_batch(obses, actions)
            print(pred_reward)
            print(pred_reward.shape)
            self.rewards[index*batch_size:last_index] = pred_reward
    
class RelabelReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, mode, device, window=1):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.true_rewards = np.empty((capacity, mode), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.window = window
        
        self.idx = 0
        self.last_save = 0
        self.current_mode = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx
    
    def set_mode(self, mode):
        self.current_mode = mode

    def add(self, obs, action, true_reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.true_rewards[self.idx], true_reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
            
    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.true_rewards[idxs], device=self.device)
        rewards = rewards[:, self.current_mode].reshape(-1,1)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max