import math
import torch
from gat import GAT
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(
        self,
        text_embedding: torch.tensor,
        structure_embedding: torch.tensor,
        node_num: torch.tensor,
        rel_num: torch.tensor
    ) -> None:
        super().__init__()
        
        self.rel_num = rel_num
        self.node_num = node_num
        self.text_embedding = text_embedding
        self.structure_embedding = structure_embedding
        self.rel_embedding = torch.nn.Embedding(rel_num, 128)
        self.linear = torch.nn.Linear(text_embedding.size(1), 512)
        # self.linear1 = torch.nn.Linear(512, 128)
        
        self.gat = GAT(
            in_dim=512,
            hidden_dim=128,
            out_dim=128,
            num_heads=2
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        bound = 6. / math.sqrt(128)
        torch.nn.init.uniform_(self.rel_embedding.weight, -bound, bound)
        F.normalize(self.rel_embedding.weight.data, p=2.0, dim=-1, out=self.rel_embedding.weight.data)  
    
    def _calc_score(
        self, 
        x: torch.tensor,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor
    ):
        head_emb = x[head_index]
        rel_emb  = self.rel_embedding(rel_type)
        tail_emb = x[tail_index]
    
        score = -(head_emb + rel_emb - tail_emb).norm(p=2.0, dim=-1) ** 2
        
        return score
    
            
    def forward(
        self,
        node_id: torch.tensor,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor,
    ):
        text_emb = self.text_embedding[node_id]
        structure_emb = self.structure_embedding[node_id]
        
        text_emb = self.linear(text_emb)
        # x = torch.concat([text_emb, structure_emb], dim=1)
        
        x = text_emb
        
        edge_index = torch.stack([head_index, tail_index])
        
        x = self.gat(
            h=x,
            edge_index=edge_index
        )
                
        return x, self.rel_embedding
    
    def loss(
        self,
        node_id: torch.tensor,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor,
        pos_head_index: torch.tensor,
        pos_rel_type: torch.tensor,
        pos_tail_index: torch.tensor,
        neg_head_index: torch.tensor,
        neg_rel_type: torch.tensor,
        neg_tail_index: torch.tensor,
    ):
        
        x, rel_emb = self.forward(
            node_id=node_id,
            head_index=head_index,
            rel_type=rel_type,
            tail_index=tail_index
        )
        
        pos_score = self._calc_score(
            x=x,
            head_index=pos_head_index,
            rel_type=pos_rel_type,
            tail_index=pos_tail_index
        )

        neg_score = self._calc_score(
            x=x,
            head_index=neg_head_index,
            rel_type=neg_rel_type,
            tail_index=neg_tail_index
        )
        
        loss = F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=2.0
        )
        
        return loss, x, rel_emb
        
        
        
        
        
        
        
        
        
        
        
        