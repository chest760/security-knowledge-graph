import math
import torch
import torch.nn.functional as F

class TransE(torch.nn.Module):
    def __init__(
        self,
        node_num: int,
        rel_num: int,
        hidden_channels: int,
        p_norm: float,
        margin: float
    ):
        super().__init__()
        self.node_emb = torch.nn.Embedding(node_num,hidden_channels)
        self.rel_emb = torch.nn.Embedding(rel_num,hidden_channels)
        self.p_norm = p_norm
        self.margin = margin
        self.hidden_channels = hidden_channels
        
        self.reset_parameters()
        
    
    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        F.normalize(self.rel_emb.weight.data, p=self.p_norm, dim=-1,
                      out=self.rel_emb.weight.data)

        
    def forward(
        self,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor,
    ):
        head_emb = self.node_emb(head_index)
        tail_emb = self.node_emb(tail_index)
        rel_emb = self.rel_emb(rel_type)
        
        head_emb = F.normalize(input=head_emb, p=self.p_norm, dim=-1)
        tail_emb = F.normalize(input=tail_emb, p=self.p_norm, dim=-1)
        
        return -((head_emb + rel_emb) - tail_emb).norm(p=self.p_norm, dim=-1)
    
    def loss(
        self,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor,
        neg_head_index: torch.tensor,
        neg_rel_type: torch.tensor,
        neg_tail_index: torch.tensor
    ) -> torch.Tensor:
        pos_score = self.forward(
                        head_index=head_index,
                        rel_type=rel_type,
                        tail_index=tail_index
                    )
        
        neg_score = self.forward(
                        head_index=neg_head_index,
                        rel_type=neg_rel_type,
                        tail_index=neg_tail_index
                    )
        
        
        loss =  F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )
        
        return loss