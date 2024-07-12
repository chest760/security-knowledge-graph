import torch
from gat import GAT
import torch.nn.functional as F

class BaseLineModel2(torch.nn.Module):
    def __init__(
        self,
        structure_node_embedding: torch.tensor,
        structure_rel_embedding: torch.tensor,
        text_node_embedding: torch.tensor
        
    ):
        super().__init__()
        self.encoder = Encoder(
            structure_node_embedding = structure_node_embedding,
            structure_rel_embedding = structure_rel_embedding,
            text_node_embedding = text_node_embedding
        )
        self.decoder = Decoder()
        
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
        
        

class Encoder(torch.nn.Module):
    def __init__(
        self,
        structure_node_embedding: torch.tensor,
        structure_rel_embedding: torch.tensor,
        text_node_embedding: torch.tensor
    ):
        super().__init__()
        self.structure_node_embedding = structure_node_embedding
        self.structure_rel_embedding = structure_rel_embedding
        self.text_node_embedding = text_node_embedding
        
        self.linear = F.Linear(text_node_embedding.size(1), 384)
        
        
    def forward(
        self,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor
    ) -> torch.tensor:
        text_node_embedding = self.linear(self.text_node_embedding)
        head_index
        pass
    
    def loss(
        self,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor,
        neg_head_index: torch.tensor,
        neg_rel_type: torch.tensor, 
        neg_tail_index: torch.tensor
    ) -> torch.tensor: 
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
        
        loss = F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )
        
        return loss

    
    def loss(
        self,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor,
        neg_head_index: torch.tensor,
        neg_rel_type: torch.tensor,
        neg_tail_index: torch.tensor,
    ) -> torch.tensor:
        pass

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor
    ) -> torch.tensor:
        pass
    
    def loss(
        self,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor,
        neg_head_index: torch.tensor,
        neg_rel_type: torch.tensor,
        neg_tail_index: torch.tensor,
    ) -> torch.tensor:
        pass
    
    
    