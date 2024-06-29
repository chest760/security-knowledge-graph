import torch
from typing import Optional
from abc import ABCMeta, abstractmethod

class TextEmbedding(torch.nn.Module, metaclass=ABCMeta):
    
    @abstractmethod()
    def __init__(self):
        super().__init__()
        pass
    
    @abstractmethod()
    def forward(
        self,
        id: torch.LongTensor,
        mask: Optional[torch.LongTensor]
    ):
        pass