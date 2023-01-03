import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

def get_layer(shape, deconv = False):
    '''
    return a layer of network
    
    '''

    if len(shape) == 2:
        in_dim, out_dim = shape
        return nn.Linear(in_dim, out_dim)
    elif len(shape) == 4:
        raise NotImplementedError(f"Conv layer is not implemented")
    else:
        return ValueError(f"illeagl shape : {shape}")

def get_act_fun(act_fun_name):
    act_fun_name = act_fun_name.lower()
    if act_fun_name == 'tanh':
        return nn.Tanh
    elif act_fun_name == 'relu':
        return nn.ReLU
    elif act_fun_name == 'sigmoid':
        return nn.Sigmoid
    elif act_fun_name == 'identity':
        return nn.Identity
    else:
        raise NotImplementedError(f"Activation function {act_fun_name} is not implemented")
    
def get_optimizer(opt_name, network, learning_rate):
    opt_name = opt_name.lower()
    if opt_name == 'sgd':
        return torch.optim.SGD(network.parameters(), lr = learning_rate)
    elif opt_name == 'adam':
        return torch.optim.Adam(network.parameters(), lr = learning_rate)
    else:
        raise NotImplementedError(f'Optimizer {opt_name}  is not supported')

class MLP(nn.Module):
    def __init__(
        self,
        in_dim : int,
        out_dim : int,
        hidden_dims : Union[int, list],
        act_fun  = 'relu',
        out_act_fun = 'identity',
        **kwargs
    ) -> None:
        super(MLP, self).__init__()
        if type[hidden_dims] == int:
            hidden_dims = [hidden_dims]
        
        hidden_dims = [in_dim] + hidden_dims
        self.networks = []
        act_fun = get_act_fun(act_fun)
        out_act_fun = get_act_fun(out_act_fun)
        
        for i in range(0, len(hidden_dims) - 1):
            input_dim, output_dim = hidden_dims[i], hidden_dims[i + 1]
            layer = get_layer([input_dim, output_dim])
            self.networks.extend([layer, act_fun()])
        out_layer = get_layer([hidden_dims[-1], out_dim])
        self.networks.extend([out_layer, out_act_fun()])
        self.networks = nn.Sequential(*self.networks)
    
    def forward(self, input):
        return self.networks(input)
    
    @property
    def weight(self):
        return [net.weight for net in self.networks if isinstance(net, nn.Linear)]