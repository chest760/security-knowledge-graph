import math
import torch
from typing import Literal
from models.gnn_models.hgt import HGT
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(
        self,
        gnn_model: HGT,
    ):
        super().__init__()
        self.gnn_model = gnn_model
        self.rel_embedding = torch.nn.Embedding(5, 256)
        
        self.reset_parameters()

    def reset_parameters(self):
        bound = 6. / math.sqrt(256)
        torch.nn.init.uniform_(self.rel_embedding.weight, -bound, bound)
        F.normalize(self.rel_embedding.weight.data, p=2.0, dim=-1,
                      out=self.rel_embedding.weight.data)
    
    def _calc_score(self, x: torch.tensor, edge_label_index: torch.tensor, edge_label: torch.tensor):
        head_emb = x[edge_label_index[0]]
        tail_emb = x[edge_label_index[1]]
        
        rel_emb = self.rel_embedding(edge_label)
        
        head_emb = F.normalize(head_emb, p=2.0, dim=-1)
        tail_emb = F.normalize(tail_emb, p=2.0, dim=-1)
        
        score = -(head_emb + rel_emb - tail_emb).norm(p=2.0, dim=-1)
        
        return score
    
    def forward(self, x_dict: dict[str, torch.tensor], edge_index_dict: dict[str, torch.tensor], edge_label: torch.tensor):
        edge_index_dict.pop(('capec', 'to', 'capec'))
        x_dict = self.gnn_model.forward(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict
        )
                
        return x_dict
    
    def loss(
        self,
        x_dict: dict[str, torch.tensor],
        edge_index_dict: dict[str, torch.tensor],
        pos_edge_label: torch.tensor,
        pos_edge_label_index: torch.tensor,
        neg_edge_label: torch.tensor,
        neg_edge_label_index: torch.tensor,
    ):
        x_dict = self.forward(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            edge_label=pos_edge_label
        )
        
        embs = x_dict["capec"]
        
        pos_score = self._calc_score(
            x=embs,
            edge_label_index=pos_edge_label_index,
            edge_label=pos_edge_label
        )

        neg_score = self._calc_score(
            x=embs,
            edge_label_index=neg_edge_label_index,
            edge_label=neg_edge_label
        )
        
        loss = F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=2.0,
        )
    
        
        return loss
        
