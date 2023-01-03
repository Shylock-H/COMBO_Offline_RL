import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from abc import ABC, abstractmethod
from typing import Union
from util.net import get_layer, get_act_fun
from util.setting import DEVICE
from torch.distributions import Categorical, Normal

class BasePolicy(ABC, nn.Module):
    def __init__(
        self,
        in_dim : int,
        action_space : gym.Space,
        hidden_dims : Union[int, list],
        act_fun = 'relu',
        *args,
        **kwargs
    ) -> None:
        super(BasePolicy, self).__init__()
        self.in_dim = in_dim
        self.act_space = action_space
        self.args = args
        self.kwargs = kwargs

        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        hidden_dims = [in_dim] + hidden_dims
        act_fun = get_act_fun(act_fun)

        self.hidden_layers = []
        for i in range(len(hidden_dims) - 1):
            input_dim, output_dim = hidden_dims[i], hidden_dims[i + 1]
            layer = get_layer([input_dim, output_dim])
            self.hidden_layers.extend([layer, act_fun()])
        
        if type(action_space) == gym.spaces.Discrete:
            self.act_dim = action_space.n
        elif type(action_space) == gym.spaces.Box:
            self.act_dim = action_space.shape[0]
        elif type(action_space) == gym.spaces.MultiBinary:
            self.act_dim = action_space.shape[0]
        else:
            raise TypeError

        
    @abstractmethod
    def forward(self, state):
        raise NotImplementedError

    @abstractmethod
    def sample(self, state, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def evaluate_action(self, states, actions, *args, **kwargs):
        raise NotImplementedError
    
class DeterministicPolicy(BasePolicy):
    def __init__(self, 
                 in_dim: int, 
                 action_space: gym.Space, 
                 hidden_dims: Union[int, list],
                 act_fun : str = 'relu',
                 out_fun : str = 'identity',
                 *args, **kwargs
                 ) -> None:
        super(DeterministicPolicy, self).__init__(in_dim, action_space, hidden_dims, act_fun, *args, **kwargs)
        
        self.deterministic = True
        
        # output layer
        out_layer = get_layer([hidden_dims[-1], self.act_dim])
        out_act_fun = get_act_fun(out_fun)
        self.networks = nn.Sequential(*self.hidden_layers, out_layer, out_act_fun())

        self.noise = torch.Tensor(self.act_dim)

        if type(action_space) != gym.spaces.Discrete:
            self.register_buffer("action_scale", 
            torch.tensor((action_space.high - action_space.low) / 2.0, dtype = torch.float,
                          device = DEVICE))
            self.register_buffer("action_bias",
            torch.tensor((action_space.high + action_space.low) / 2.0, dtype = torch.float,
                          device = DEVICE))
        else:
            raise TypeError('Deterministic Policy is not supported for discrete action space!')
        
    def forward(self, state):
        return self.networks(state)
    
    def sample(self, states):
        action_pre = self.networks(states)
        action_raw = torch.tanh(action_pre)
        action_scaled = action_raw * self.action_scale + self.action_bias

        return {
            'action_pre' : action_pre,
            'action_raw' : action_raw,
            'action_scaled' : action_scaled
        }
    
    def evaluate_action(self, states, actions, *args, **kwargs):
        
        return self.sample(states)
    
class CategoricalPolicy(BasePolicy):
    def __init__(self, 
                 in_dim: int, 
                 action_space: gym.Space, 
                 hidden_dims: Union[int, list], 
                 act_fun : str = 'relu', 
                 out_fun : str =  'identity',
                 *args, **kwargs) -> None:
        super(CategoricalPolicy, self).__init__(in_dim, action_space, hidden_dims, act_fun, *args, **kwargs)

        self.deterministic = False

        # output layer
        out_layer = get_layer([hidden_dims[-1], self.act_dim])
        out_act_fun = get_act_fun(out_fun)
        self.networks = nn.Sequential(*self.hidden_layers, out_layer, out_act_fun())

    def forward(self, state):
        
        return self.networks(state)

    def sample(self, state, deterministic = False):
        logit = self.forward(state)
        probs = torch.softmax(logit, dim = -1)
       
        if deterministic:
            return {
                'logit' : logit,
                'prob' : probs,
                'action' : torch.argmax(probs, dim = -1, keepdim = True),
                'log_prob' : torch.log(torch.max(probs, dim = -1, keepdim = True)[0] + 1e-6)
            }
        else :
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return {
                'logit' : logit,
                'prob' : probs,
                'action' : action.view(-1, 1),
                'log_prob' : log_prob.view(-1, 1)
            }
    
    def evaluate_action(self, states, actions, *args, **kwargs):
        if len(actions.shape) == 2:
            actions = actions.view(-1)

        logit = self.forward(states)
        probs = torch.softmax(logit, dim = -1)
        dist = Categorical(probs)

        return {
            'log_prob' : dist.log_prob(actions),
            'entropy' : dist.entropy()
        }

class GaussPolicy(BasePolicy):
    def __init__(self, 
                 in_dim: int, 
                 action_space: gym.Space, 
                 hidden_dims: Union[int, list], 
                 act_fun : str = 'relu',
                 out_fun : str = 'identity',
                 out_std : bool = True,
                 conditioned_std : bool = True,
                 reparameter : bool = True,
                 log_std : float = None,
                 log_std_min : float = -20.0,
                 log_std_max : float = 2.0,
                 stable_log_prob : bool = True,
                 *args, **kwargs) -> None:
        super().__init__(in_dim, action_space, hidden_dims, act_fun, *args, **kwargs)

        self.deterministic = False

        self.out_std = out_std
        self.reparemeter = reparameter

        if self.out_std : 
            out_layer = get_layer([hidden_dims[-1], 2 * self.act_dim])
        else:
            out_layer = get_layer(hidden_dims[-1], self.act_dim)

        out_act_fun = get_act_fun(out_fun)
        self.networks = nn.Sequential(*self.hidden_layers, out_layer, out_act_fun())

        if type(action_space) != gym.spaces.Discrete:
            self.register_buffer("action_scale", 
            torch.tensor((action_space.high - action_space.low) / 2.0, dtype = torch.float,
                          device = DEVICE))
            self.register_buffer("action_bias",
            torch.tensor((action_space.high + action_space.low) / 2.0, dtype = torch.float,
                          device = DEVICE))
        else:
            raise TypeError('Gauss Policy is not supported for discrete action space!')
        
        if log_std == None:
            self.log_std = -0.5 * np.ones(self.act_dim, dtype = np.float32)
        else:
            self.log_std = log_std
        
        if conditioned_std:
            self.log_std = nn.Parameter(torch.as_tensor(self.log_std)).to(DEVICE)
        else:
            self.log_std = torch.tensor(self.log_std, dtype = torch.float, device = DEVICE)
        
        self.register_buffer("log_std_min", torch.tensor(log_std_min, dtype = torch.float, device = DEVICE))
        self.register_buffer("log_std_max", torch.tensor(log_std_max, dtype = torch.float, device = DEVICE))

        self.stable_log_prob = stable_log_prob

    def forward(self, state):
        out = self.networks(state)
        mu = out[ : , : self.act_dim]
        if self.out_std:
            action_log_std = out[ : , self.act_dim : ]
        else:
            action_log_std = self.log_std
        
        return mu, action_log_std
    
    def sample(self, state, deterministic = False):
        mu, log_std = self.forward(state)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max).expand_as(mu)

        dist = Normal(mu, log_std.exp())

        if deterministic:
            action_no_tanh = mu.detach()
        else:
            if self.reparemeter:
                action_no_tanh = dist.rsample()
            else:
                action_no_tanh = dist.sample()
        
        action_raw = torch.tanh(action_no_tanh)
        action_scaled = self.action_scale * action_raw + self.action_bias

        log_prob_no_tanh = dist.log_prob(action_no_tanh)
        if self.stable_log_prob:
            log_prob = log_prob_no_tanh - (
                    2 * (np.log(2) - action_no_tanh - F.softplus(-2 * action_no_tanh)))
        else:
            log_prob = log_prob_no_tanh
        log_prob = torch.sum(log_prob, dim = -1, keepdim = True)

        return {
            'action_raw' : action_raw,
            'action' : action_scaled,
            'log_prob' : log_prob,
            'log_std' : log_std
        }
    
    def evaluate_action(self, states, actions):
        mu, log_std = self.forward(states)
        dist = Normal(mu, log_std.exp())
        
        log_prob = dist.log_prob(actions)
        log_prob = log_prob.sum(dim = -1, keepdim = True)
        entropy = dist.entropy().sum(dim = -1, keepdim = True)

        return {
            'log_prob' : log_prob,
            'entropy' : entropy
        }
