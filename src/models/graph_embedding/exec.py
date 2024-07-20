import torch
from typing import Literal
from transe import transE
from transh import transH
import torch.nn.functional as F
from utils.triplet_loader import TripletLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def data_loader(
    triplet: torch.tensor
):
    loader = TripletLoader(
        
    )

class Exec:
    def __init__(
        self,
        model_type: Literal["transh", "transe"],
        node_num: int,
        relaion_num: int,
        hidden_channels: int,
        p_norm: float = 2.0,
        margin: float = 2.0
    ) -> None:
        if model_type == "transe":
            self.model = transE(
                node_num=node_num,
                relation_num=relaion_num,
                hidden_channels=hidden_channels
            )
        elif model_type == "transh":
            self.model = transH(
                node_num=node_num,
                relation_num=relaion_num,
                hidden_channels=hidden_channels
            )
        self.margin = margin
        self.p_norm = p_norm
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
    
    def train(
        self,
        positive_triplet: torch.tensor,
        negative_triplet: torch.tensor
    ):
        pos_score = self.model.forward(
            head_index=positive_triplet[:, 0],
            rel_type=positive_triplet[:, 1],
            tail_index=positive_triplet[:, 2],
        )
        
        neg_score = self.model.forward(
            head_index=negative_triplet[:, 0],
            rel_type=negative_triplet[:, 1],
            tail_index=negative_triplet[:, 2],
        )
        
        loss = F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )
        
        loss.backward()
        
        
        
        
    
    @torch.nograd()
    def valid(self):
        pass
    
    @torch.nograd()
    def test(self):
        pass 
    
    
    def run(
        self,
        train_positive_triplet: torch.tensor,
        train_negative_triplet: torch.tensor,
        valid_positive_triplet: torch.tensor,
        valid_negative_triplet: torch.tensor,
        test_positive_triplet: torch.tensor,
        test_negative_triplet: torch.tensor,
    ):
        for epoch in range(1, 201):
            train_loss = self.train()
            valid_loss = self.valid()
            if epoch % 30 == 0:
                self.test()
            
            