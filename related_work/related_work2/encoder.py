import torch
from transe import TransE
from word2vec import Word2vecModel

class Encoder(torch.nn.Module):
    def __init__(
        self,
        node_num: int,
        rel_num: int,
        hidden_channels: int,
        p_norm: float,
        margin: float
    ):
        super().__init__()
        
        self.transe = TransE(
                    node_num=node_num,
                    rel_num=rel_num,
                    hidden_channels=hidden_channels,
                    p_norm=p_norm,
                    margin=margin
                )
        
        self.word2vec = Word2vecModel()
    
    def forward(self):
        pass
    
    
    def loss(self):
        pass