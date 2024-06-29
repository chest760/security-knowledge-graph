import math
import torch
import torch.nn.functional as F
from base import GraphEmbedding

class transH(GraphEmbedding):
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
        
        # 変換ベクトル
        self.d_r_emb = torch.nn.Embedding(relation_num, hidden_channels)
        
        # 法線ベクトル
        self.w_r_emb = torch.nn.Embedding(relation_num, hidden_channels)
        self.p_norm = p_norm
        self.margin = margin
    
    def reset_parmeters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.relation_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.d_r_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.w_r_emb.weight, -bound, bound)
        F.normalize(self.d_r_emb.weight.data, p=self.p_norm, dim=-1,
                      out=self.d_r_emb.weight.data)
        F.normalize(self.w_r_emb.weight.data, p=self.p_norm, dim=-1,
                      out=self.w_r_emb.weight.data)
        
    
    def forward(
        self,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor
    ) -> torch.Tensor:
        head = self.node_emb(head_index)
        tail = self.node_emb(tail_index)

        
        d_r_emb = self.d_r_emb(rel_type)
        w_r_rmb = F.normalize(self.w_r_emb(rel_type), p=self.p_norm, dim=-1)
        
        h_proj = head - (w_r_rmb * head).sum(dim=1).unsqueeze(dim=1) * w_r_rmb
        t_proj = tail - (w_r_rmb * tail).sum(dim=1).unsqueeze(dim=1) * w_r_rmb
        
        h_proj = F.normalize(h_proj, p=self.p_norm, dim=-1)
        t_proj = F.normalize(t_proj, p=self.p_norm, dim=-1)
        
        score = -((h_proj + d_r_emb - t_proj ).norm(p=self.p_norm, dim=-1))**2
        
        return score
        

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

        