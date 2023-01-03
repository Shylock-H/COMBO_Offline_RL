import torch
from abc import ABC, abstractmethod
from collections import namedtuple
import gym
import numpy as np
import warnings
from util.setting import DEVICE

Transition = namedtuple('Transition', ['obs', 'action', 'reward', 'next_obs', 'done'])

class BaseBuffer(ABC):
    
    def __init__(self):
        raise NotImplementedError
    
    @abstractmethod
    def add_transition(self, obs, action, reward, next_obs, done):
        raise NotImplementedError
    
    @abstractmethod
    def add_traj(self, obs_list, action_list, reward_list, next_obs_list, done_list):
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size):
        raise NotImplementedError


class ReplayBuffer(BaseBuffer):
    '''
    basic buffer used in DQN and so on
    '''
    def __init__(self, obs_space, action_space, buffer_size = 100000, **kwargs):
        self.buf_size = buffer_size
        self.cur = 0
        self.obs_space = obs_space
        self.action_space = action_space
        self.obs_dim = obs_space.shape[0]

        if type(action_space) == gym.spaces.discrete.Discrete:
            self.action_dim = 1
            self.discrete_action = True
        elif type(action_space) == gym.spaces.box.Box:
            self.action_dim = action_space.shape[0]
            self.discrete_action = False
        else:
            raise NotImplementedError('Not support type for action space')
        
        self.obs_buf = np.zeros((self.buf_size, self.obs_dim))
        self.next_obs_buf = np.zeros((self.buf_size, self.obs_dim))
        self.action_buf = np.zeros((self.buf_size, self.action_dim))
        self.reward_buf = np.zeros((self.buf_size, 1))
        self.done_buf = np.zeros((self.buf_size, 1))

        self.allow_size = 0

    def add_transition(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.cur] = obs
        self.action_buf[self.cur] = action
        self.reward_buf[self.cur] = reward
        self.next_obs_buf[self.cur] = next_obs
        self.done_buf[self.cur] = done

        self.cur = (self.cur + 1) % self.buf_size
        self.allow_size = min(self.allow_size + 1, self.buf_size)

    def add_traj(self, obs_list, action_list, reward_list, next_obs_list, done_list):
        for obs, action, reward, next_obs, done in zip(obs_list, action_list, reward_list, next_obs_list, done_list):
            self.add_transition(obs, action, reward, next_obs, done)

    def add_batch(self, obs, action, reward, next_obs, done):
        batch_size = obs.shape[0]
        if type(obs) == torch.Tensor:
            obs = obs.cpu().numpy()
        if type(next_obs) == torch.Tensor:
            next_obs = next_obs.cpu().numpy()
        if type(action) == torch.tensor:
            action = action.cpu().numpy()
        if type(reward) == torch.Tensor:
            reward = reward.cpu().numpy()
        if type(done) == torch.Tensor:
            done = done.cpu().numpy()

        for idx in range(batch_size):
            self.add_transition(obs[idx], action[idx], reward[idx], next_obs[idx], done[idx])

    def sample(self, batch_size, to_tensor = True, seq = False, duplicate = False):
        if not duplicate and batch_size > self.allow_size:
            warnings.warn("Sampling size is larger than buffer size")
            batch_size = min(self.allow_size, batch_size)
        
        if seq:
            start_idx = np.random.choice(range(self.allow_size))
            idx = []
            for i in range(batch_size):
                idx.append((start_idx + i) % self.buf_size)
        else:
            idx = np.random.choice(range(self.allow_size), size = batch_size, replace = duplicate)
        
        batch_obs, batch_act, batch_reward, batch_next_obs, batch_done = self.obs_buf[idx], self.action_buf[idx], \
                                                                         self.reward_buf[idx],  self.next_obs_buf[idx], \
                                                                         self.done_buf[idx]
        if to_tensor:
            batch_obs = torch.FloatTensor(batch_obs).to(DEVICE)
            if self.discrete_action:
                batch_act = torch.LongTensor(batch_act).to(DEVICE)
            else:
                batch_act = torch.FloatTensor(batch_act).to(DEVICE)
            batch_reward = torch.FloatTensor(batch_reward).to(DEVICE)
            batch_next_obs = torch.FloatTensor(batch_next_obs).to(DEVICE)
            batch_done = torch.FloatTensor(batch_done).to(DEVICE)
        
        return dict(
            obs = batch_obs,
            action = batch_act,
            reward = batch_reward,
            next_obs = batch_next_obs,
            done = batch_done
        )
    
    def load_dataset(self, dataset):
        print('\033[1;34m Loading dataset ...\033[0m : ')
        observations = np.array(dataset["observations"])
        next_observations = np.array(dataset["next_observations"])
        actions = np.array(dataset["actions"])
        rewards = np.array(dataset["rewards"]).reshape(-1, 1)
        dones = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.obs_buf = observations
        self.action_buf = actions
        self.reward_buf = rewards
        self.next_obs_buf = next_observations
        self.done_buf = dones

        self.allow_size = self.buf_size
        self.cur = self.buf_size
        
        print('\033[1;34m Loading finished \033[0m')
