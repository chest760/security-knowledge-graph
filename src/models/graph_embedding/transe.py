import math
import torch
import torch.nn.functional as F
from base import GraphEmbedding

class transE(GraphEmbedding):
    def __init__(self, 
                 node_num: int, 
                 relation_num: int, 
                 hidden_channels: int,
                 margin: float = 1.0,
                 p_norm: float = 2.0,
                ):
        super().__init__(
            node_num, 
            relation_num, 
            hidden_channels
        )
        
        self.margin = margin
        self.p_norm = p_norm
    
    
    def reset_parmeters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.relation_emb.weight, -bound, bound)
        F.normalize(self.relation_emb.weight.data, p=self.p_norm, dim=-1,
                      out=self.rel_emb.weight.data)
    

    def forward(
        self,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
    ) -> torch.Tensor:

        head = self.node_emb(head_index)
        rel = self.relation_emb(rel_type)
        tail = self.node_emb(tail_index)

        head = F.normalize(head, p=self.p_norm, dim=-1)
        tail = F.normalize(tail, p=self.p_norm, dim=-1)

        return -((head + rel) - tail).norm(p=self.p_norm, dim=-1)
    
    
    def loss(
        self,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
        neg_head_index: torch.Tensor,
        neg_rel_type: torch.Tensor,
        neg_tail_index: torch.Tensor
    ) -> torch.Tensor:

        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(neg_head_index, neg_rel_type, neg_tail_index)

        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )

        
    
