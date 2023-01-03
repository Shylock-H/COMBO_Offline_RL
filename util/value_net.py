import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import Union
from util.net import get_act_fun, get_layer

class BaseValue(ABC, nn.Module):
    def __init__(
        self,
        in_dim : int,
        out_dim : int,
        hidden_dims : Union[int, list],
        act_fun : str = 'relu',
        *args,
        **kwargs
    ) -> None:
        super(BaseValue, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
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
    
    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

class QNet(BaseValue):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        hidden_dims: Union[int, list], 
        act_fun : str = 'relu', 
        out_act_fun : str = 'identity',
        *args, 
        **kwargs
    ) -> None:
        super(QNet, self).__init__(in_dim, out_dim, hidden_dims, act_fun, *args, **kwargs)
        out_layer = get_layer([hidden_dims[-1], self.out_dim])
        out_act_fun = get_act_fun(out_act_fun)
        self.networks = nn.Sequential(*self.hidden_layers, out_layer, out_act_fun())

    def forward(self, input):
        return self.networks(input)

