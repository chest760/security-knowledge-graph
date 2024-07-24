import math
import torch
from .base import GraphEmbedding
import torch.nn.functional as F

class RotatE(GraphEmbedding):
    def __init__(
        self, 
        node_num: int, 
        relation_num: int, 
        hidden_channels: int,
        margin: float = 1.0,
        p_norm: float = 2.0
    ):
        super().__init__(
            node_num, 
            relation_num, 
            hidden_channels
        )
        self.margin = margin
        self.p_norm = p_norm
        self.node_emb_im = torch.nn.Embedding(node_num, hidden_channels)
        self.rel_emb = torch.nn.Embedding(relation_num, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb.weight)
        torch.nn.init.xavier_uniform_(self.node_emb_im.weight)
        torch.nn.init.uniform_(self.rel_emb.weight, 0, 2 * math.pi)
        
    def forward(
        self,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
    ) -> torch.Tensor:

        head_re = self.node_emb(head_index)
        head_im = self.node_emb_im(head_index)
        tail_re = self.node_emb(tail_index)
        tail_im = self.node_emb_im(tail_index)

        rel_theta = self.rel_emb(rel_type)
        rel_re, rel_im = torch.cos(rel_theta), torch.sin(rel_theta)

        re_score = (rel_re * head_re - rel_im * head_im) - tail_re
        im_score = (rel_re * head_im + rel_im * head_re) - tail_im
        complex_score = torch.stack([re_score, im_score], dim=2)
        score = torch.linalg.vector_norm(complex_score, dim=(1, 2))

        return self.margin - score
    
    def loss(
        self,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
        neg_head_index: torch.Tensor,
        neg_rel_type: torch.Tensor,
        neg_tail_index: torch.Tensor
    ) -> torch.Tensor:

        pos_score = self.forward(head_index, rel_type, tail_index)
        neg_score = self.forward(neg_head_index, neg_rel_type, neg_tail_index)
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        neg_target = torch.zeros_like(neg_score)
        target = torch.cat([pos_target, neg_target], dim=0)

        return F.binary_cross_entropy_with_logits(scores, target)