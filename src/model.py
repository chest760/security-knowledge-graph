import math
import torch
from typing import Literal, Union
from models.gnn_models.hgt import HGT
from models.gnn_models.gat import GAT
from models.gnn_models.rgat import RGAT
import torch.nn.functional as F
from utils.static_seed import static_seed

static_seed(42)

class Model(torch.nn.Module):
    def __init__(
        self,
        gnn_model: Union[HGT, GAT, RGAT],
    ):
        super().__init__()
        self.gnn_model = gnn_model
        self.rel_embedding = torch.nn.Embedding(5, 256)
        
        self.reset_parameters()

    def reset_parameters(self):
        bound = 6. / math.sqrt(256)
        torch.nn.init.uniform_(self.rel_embedding.weight, -bound, bound)
        F.normalize(self.rel_embedding.weight.data, p=2.0, dim=-1, out=self.rel_embedding.weight.data)
    
    def _calc_score(self, x: torch.tensor, edge_label_index: torch.tensor, edge_label: torch.tensor):
        head_emb = x[edge_label_index[0]]
        tail_emb = x[edge_label_index[1]]
        
        rel_emb = self.rel_embedding(edge_label)
        
        head_emb = F.normalize(head_emb, p=2.0, dim=-1)
        tail_emb = F.normalize(tail_emb, p=2.0, dim=-1)
        
        score = -(head_emb + rel_emb - tail_emb).norm(p=2.0, dim=-1)
        
        return score
    
    def forward(self, x_dict: dict[str, torch.tensor], edge_index_dict: dict[str, torch.tensor], edge_label: torch.tensor):
        # edge_index = edge_index_dict.pop(('capec', 'to', 'capec'))
        
        if isinstance(self.gnn_model, (HGT, GAT)):
            x_dict = self.gnn_model.forward(
                x_dict=x_dict,
                edge_index_dict=edge_index_dict
            )
            
        
        # elif isinstance(self.gnn_model, RGAT):
        #     x_dict = self.gnn_model.forward(
        #         x_dict=x_dict,
        #         edge_index=edge_index,
        #         edge_label=edge_label
        #     )
                
        return x_dict
    
    def loss(
        self,
        x_dict: dict[str, torch.tensor],
        edge_index_dict: dict[str, torch.tensor],
        edge_label: torch.tensor,
        pos_edge_label: torch.tensor,
        pos_edge_label_index: torch.tensor,
        neg_edge_label: torch.tensor,
        neg_edge_label_index: torch.tensor,
    ):
        x_dict = self.forward(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            edge_label=edge_label
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
        
        # loss = F.margin_ranking_loss(
        #     pos_score,
        #     neg_score,
        #     target=torch.ones_like(pos_score),
        #     margin=3.0,
        # )
        
        # return loss
        
    
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        neg_target = torch.zeros_like(neg_score)
        target = torch.cat([pos_target, neg_target], dim=0)
    

        return F.binary_cross_entropy_with_logits(scores, target)
        

        
