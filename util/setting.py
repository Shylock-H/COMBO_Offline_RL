import numpy as np
import torch
import random
import gym

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_PATH = 'log'

def set_device():
    global DEVICE

    if torch.cuda.is_available():
        DEVICE = 'cuda'

    print(f'Setting device : {DEVICE}')

def set_global_seed(seed = 13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # gym.Env.reset(seed = seed)