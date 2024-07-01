import torch
from abc import ABCMeta, abstractmethod

class GraphEmbedding(torch.nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        node_num: int,
        relation_num: int,
        hidden_channels: int
    ):
        super().__init__()
        self.node_emb = torch.nn.Embedding(node_num, hidden_channels)
        self.relation_emb = torch.nn.Embedding(relation_num, hidden_channels)
        self.hidden_channels = hidden_channels
    
    @abstractmethod
    def reset_parmerters(self):
        pass
    
    @abstractmethod
    def forward(
        head_index:torch.Tensor,
        rel_type:torch.Tensor,
        tail_index: torch.Tensor
    ) -> torch.Tensor:
        pass
        