import math
import torch
from gat import GAT
import torch.nn.functional as F
from convkb import ConvKB
from src.utils.static_seed import static_seed

static_seed(42)

class BaseLineModel2(torch.nn.Module):
    def __init__(
        self,
        structure_node_embedding: torch.tensor,
        text_node_embedding: torch.tensor,
        node_num: int,
        rel_num: int
        
    ):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(node_num=node_num, rel_num=rel_num)
        
        self.structure_node_embedding = structure_node_embedding
        self.structure_rel_embedding = torch.nn.Embedding(rel_num, 100)
        self.text_node_embedding = text_node_embedding
        self.linear = torch.nn.Linear(text_node_embedding.size(1), 384)
        
        self.node_num = node_num
        self.rel_num  = rel_num
        
        self.reset_parameters()

    def reset_parameters(self):
        bound = 6. / math.sqrt(100)
        torch.nn.init.uniform_(self.structure_rel_embedding.weight, -bound, bound)
        F.normalize(self.structure_rel_embedding.weight.data, p=2.0, dim=-1, out=self.structure_rel_embedding.weight.data)  
        
    def forward(
        self,
        node_id: torch.tensor, 
        head_index: torch.tensor, 
        rel_type: torch.tensor, 
        tail_index: torch.tensor, 
        head_label_index: torch.tensor,
        rel_label_type: torch.tensor, 
        tail_label_index: torch.tensor,
    ):
        
        # concat text embedding and structure embedding
        text_emb = self.text_node_embedding[node_id]
        structure_emb = self.structure_node_embedding[node_id]
        
        
        text_emb = self.linear(text_emb) # Text Embedding size -> 384
        x = torch.concat([structure_emb, text_emb], dim=1) # x size: 128 + 384 -> 512
        
        # input gat model # updated_x size: 100
        updated_x = self.encoder.forward(
            x=x,
            head_index=head_index,
            rel_type=rel_type,
            tail_index=tail_index
        )
        
        # calcurate the score
        encoder_score = self.encoder.calc_score(
            x=updated_x, 
            rel_emb=self.structure_rel_embedding,
            head_index=head_label_index,
            rel_type=rel_label_type,
            tail_index=tail_label_index
        )
        
        
        # input convKB model and calcurate score
        decoder_score, l2 = self.decoder.forward(
            updated_x,
            rel_emb = self.structure_rel_embedding,
            head_index=head_label_index,
            rel_type=rel_label_type,
            tail_index=tail_label_index
        )
        
        return encoder_score, decoder_score, l2
        
        

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()        
        self.gat = GAT(
            in_dim=512,
            hidden_dim=128,
            out_dim=128,
            num_heads=2
        )
        self.linear = torch.nn.Linear(128, 100)
        
    def forward(
        self,
        x: torch.tensor,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor
    ) -> torch.tensor:
        
        x = self.gat.forward(
            h=x,
            edge_index=torch.stack([head_index, tail_index])
        )
        
        x = self.linear(x)
        return x
    
    def calc_score(
        self, 
        x: torch.tensor,
        rel_emb: torch.tensor,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor
    ):
        head_emb = x[head_index]
        rel_emb  = rel_emb(rel_type)
        tail_emb = x[tail_index]
        
        head_emb = F.normalize(head_emb, p=2.0, dim=-1)
        tail_emb = F.normalize(tail_emb, p=2.0, dim=-1)
        score =  -(head_emb + rel_emb - tail_emb).norm(p=2.0, dim=-1)
        return score
    

class Decoder(torch.nn.Module):
    def __init__(
        self,
        node_num: int,
        rel_num: int
    ):
        super().__init__()
        
        self.convkb = ConvKB(
            node_num=node_num,
            rel_num=rel_num,
            kernel_size=1,
            hidden_channels=100,
            out_channels=100
        )
    
    def forward(
        self,
        x: torch.tensor,
        rel_emb: torch.tensor,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor
    ) -> torch.tensor:
        score, l2 = self.convkb.forward(
            x=x,
            rel_emb=rel_emb,
            head_index=head_index,
            rel_type=rel_type,
            tail_index=tail_index
        )
        
        return score, l2
    
    
