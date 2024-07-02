import math
import torch
import torch.nn.functional as F
from word2vec import Word2vecModel

class TextEmbeddingCNN(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 300,
        kernel_size: int = 2
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels=input_channels, 
            out_channels=100, 
            kernel_size=kernel_size
        ) 
        
        self.conv2 = torch.nn.Conv1d(
            in_channels=100, 
            out_channels=100, 
            kernel_size=kernel_size
        ) 
        
        self.max_pooling = torch.nn.MaxPool1d(kernel_size=kernel_size)
        self.mean_pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.relu = torch.nn.ReLU()
    
    def forwad(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        x = x.permute(0, 2, 1) 
        x = self.conv1(x) 
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2(x) 
        x = self.relu(x)
        x = self.mean_pooling(x)  
        x = x.squeeze(2)  
        return x
        

class Model(torch.nn.Module):
    def __init__(
        self,
        node_num: int,
        rel_num: int,
        hidden_channels: int,
        pre_embedding: torch.Tensor,
        p_norm: float=2.0,
        margin: float=1.0,
        ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.node_emb_d = pre_embedding
        self.node_emb_s = torch.nn.Embedding(node_num, hidden_channels)
        self.w_r_emb = torch.nn.Embedding(rel_num, 100)
        self.d_r_emb = torch.nn.Embedding(rel_num, 100)
        self.conv = TextEmbeddingCNN()
        self.p_norm = p_norm
        self.margin = margin
        self.reset_parameters()
    
    def reset_parameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb_s.weight, -bound, bound)
        torch.nn.init.uniform_(self.d_r_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.w_r_emb.weight, -bound, bound)
        F.normalize(self.d_r_emb.weight.data, p=self.p_norm, dim=-1, out=self.d_r_emb.weight.data)
        F.normalize(self.w_r_emb.weight.data, p=self.p_norm, dim=-1, out=self.w_r_emb.weight.data)

    def forward(
        self,
        head: torch.tensor,
        rel: torch.tensor,
        tail: torch.tensor
    ):
        head_d: torch.tensor = self.conv.forwad(self.node_emb_d[head])
        tail_d: torch.tensor = self.conv.forwad(self.node_emb_d[tail])
        # head_d = text_embedding[head]
        # tail_d = text_embedding[tail]
        
        head_s = self.node_emb_s(head)
        tail_s = self.node_emb_s(tail)

        d_r_emb = self.d_r_emb(rel)
        w_r_rmb = F.normalize(self.w_r_emb(rel), p=self.p_norm, dim=-1)
        
        head_s = F.normalize(head_s, p=self.p_norm, dim=-1)
        tail_s = F.normalize(tail_s, p=self.p_norm, dim=-1)


        
        proj_head_d = head_d - (w_r_rmb * head_d).sum(dim=1).unsqueeze(dim=1) * w_r_rmb
        proj_tail_d = tail_d - (w_r_rmb * tail_d).sum(dim=1).unsqueeze(dim=1) * w_r_rmb
        
        proj_head_s = head_s - (w_r_rmb * head_s).sum(dim=1).unsqueeze(dim=1) * w_r_rmb
        proj_tail_s = tail_s - (w_r_rmb * tail_s).sum(dim=1).unsqueeze(dim=1) * w_r_rmb
        
        f_ss =  -((proj_head_s + d_r_emb - proj_tail_s).norm(p=self.p_norm, dim=-1)) ** 2
        f_dd =  -((proj_head_d + d_r_emb - proj_tail_d).norm(p=self.p_norm, dim=-1)) ** 2
        f_sd =  -((proj_head_s + d_r_emb - proj_tail_d).norm(p=self.p_norm, dim=-1)) ** 2
        f_ds =  -((proj_head_d + d_r_emb - proj_tail_s).norm(p=self.p_norm, dim=-1)) ** 2
        
        f_r = f_ss + f_dd + f_sd + f_ds
    
        return f_r
    
    
    
    def loss(
        self,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
        neg_head_index: torch.Tensor,
        neg_rel_type: torch.Tensor,
        neg_tail_index: torch.Tensor,
        valid :bool = False
    ) -> torch.Tensor:

        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(neg_head_index, neg_rel_type, neg_tail_index)
        if valid:
            print(f"POS : {pos_score}")
            print(f"NEG : {neg_score}")

        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )
        