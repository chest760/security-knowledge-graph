import math
import torch
import torch.nn.functional as F

class TransH(torch.nn.Module):
    def __init__(
        self,
        node_num: int,
        rel_num: int,
        hidden_channels: int,
        p_norm: float,
        margin: float
        
    ):
        super().__init__()
        
        self.node_emb = torch.nn.Embedding(node_num, hidden_channels)
        self.rel_emb = torch.nn.Embedding(rel_num, hidden_channels)
        self.w_rel_emb = torch.nn.Embedding(rel_num, hidden_channels)
        self.p_norm = p_norm
        self.margin = margin
        self.hidden_channels = hidden_channels
        
    def reset_prameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.w_rel_emb.weight, -bound, bound)
        F.normalize(self.rel_emb.weight.data, p=self.p_norm, dim=-1, out=self.rel_emb.weight.data)
        F.normalize(self.w_rel_emb.weight.data, p=self.p_norm, dim=-1, out=self.w_rel_emb.weight.data)
    
    def forward(
        self,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor
    ) -> torch.tensor:
        head_emb = self.node_emb(head_index)
        tail_emb = self.node_emb(tail_index)
        rel_emb = self.rel_emb(rel_type)
        
        w_rel_rmb = F.normalize(self.w_rel_emb(rel_type), p=self.p_norm, dim=-1)
        head_emb = F.normalize(head_emb, p=self.p_norm, dim=-1)
        tail_emb = F.normalize(tail_emb, p=self.p_norm, dim=-1)
        
        h_proj = head_emb - (w_rel_rmb * head_emb).sum(dim=1).unsqueeze(dim=1) * w_rel_rmb
        t_proj = tail_emb - (w_rel_rmb * tail_emb).sum(dim=1).unsqueeze(dim=1) * w_rel_rmb
        

        score = -((h_proj + rel_emb - t_proj ).norm(p=self.p_norm, dim=-1))**2
        
        return score
    
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
    

class TransHTrain:
    def __init__(
        self,
        node_num: int,
        rel_num: int,
        hidden_channels: int,
        p_norm: float,
        margin: float
    ) -> None:
        self.model = TransH(
            node_num=node_num,
            rel_num=rel_num,
            hidden_channels=hidden_channels,
            p_norm=p_norm,
            margin=margin
        )
        
        self.node_num = node_num
        self.rel_num = rel_num
        self.hidden_channels = hidden_channels
    
    
    def run(
        self,
        train_triplet: torch.tensor,
        valid_triplet: torch.tensor,
        test_triplet: torch.tensor
    ):
        for epoch in range(1, 301):
            train_loss = self.train()
            valid_loss = self.valid()
            print(f"Train Loss: {train_loss}, Valid Loss: {valid_loss}")
            
            if epoch % 20 == 0:
                self.test()    

    
    def train(self):
        self.model.train()
        
         