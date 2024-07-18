import torch
from convkb import ConvKB
import torch.nn.functional as F

class Decoder(torch.nn.Module):
    def __init__(
        self,
        node_num :int,
        rel_num: int,
        kernel_size: int,
        hidden_channels: int,
        out_channels: int        
    ):
        super().__init__()
        
        self.convkb = ConvKB(
            node_num=node_num,
            rel_num=rel_num,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            out_channels=out_channels
        )
        
    def forward(
        self,
        x: torch.tensor,
        rel_emb: torch.tensor,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor
    ):
        score, l2 = self.convkb.forward(
            x=x,
            rel_emb=rel_emb,
            head_index=head_index,
            rel_type=rel_type,
            tail_index=tail_index
        )
        
        return score, l2
        
        
    
    def loss(
        self,
        x: torch.tensor,
        rel_emb: torch.tensor,
        pos_head_index: torch.tensor,
        pos_rel_type: torch.tensor,
        pos_tail_index: torch.tensor,
        neg_head_index: torch.tensor,
        neg_rel_type: torch.tensor,
        neg_tail_index: torch.tensor,
    ):
        pos_score, pos_l2 = self.forward(
            x=x,
            rel_emb=rel_emb,
            head_index=pos_head_index,
            rel_type=pos_rel_type,
            tail_index=pos_tail_index
        )
        
        neg_score, neg_l2 = self.forward(
            x=x,
            rel_emb=rel_emb,
            head_index=neg_head_index,
            rel_type=neg_rel_type,
            tail_index=neg_tail_index
        )

        score = torch.cat([pos_score, neg_score])
        y = torch.cat([torch.ones_like(pos_score), -1 * torch.ones_like(neg_score)]) 
        l2 = (pos_l2 + neg_l2) / 2
        
        loss = F.soft_margin_loss(score, y, reduction="mean") + 0.2 * l2
        
        return loss
        
        