from abc import ABC, abstractmethod
import torch


class BaseAgent(ABC):

    def __init__(self, **kwargs) -> None:
        
        self.networks = {}
    
    @abstractmethod
    def update(self, batch_data):
        raise NotImplementedError
    
    @abstractmethod
    def choose_action(self, state, deterministic = True):
        raise NotImplementedError

