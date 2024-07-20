import math
import torch
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        kernel_size: int
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


class BaseLineModel1(torch.nn.Module):
    def __init__(
        self,
        node_num: int,
        rel_num: int,
        node_dim: int,
        text_embedding: torch.tensor,
        text_embedding_dim: int,
        margin: float,
        p_norm: float
    ):
        super().__init__()
        
        self.node_emb_s = torch.nn.Embedding(node_num, node_dim)
        self.node_emb_d = text_embedding
        self.w_rel_emb = torch.nn.Embedding(rel_num, node_dim)
        self.rel_emb = torch.nn.Embedding(rel_num, node_dim)
        self.node_dim = node_dim
        self.margin = margin
        self.p_norm = p_norm
        
        self.cnn = CNN(
            input_channels=text_embedding_dim,
            kernel_size=2
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        bound = 6. / math.sqrt(self.node_dim)
        torch.nn.init.uniform_(self.node_emb_s.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.w_rel_emb.weight, -bound, bound)
        F.normalize(self.rel_emb.weight.data, p=self.p_norm, dim=-1, out=self.rel_emb.weight.data)
        F.normalize(self.w_rel_emb.weight.data, p=self.p_norm, dim=-1, out=self.w_rel_emb.weight.data)
    
    
    def forward(
        self,
        head_index: torch.tensor,
        rel_index: torch.tensor,
        tail_index: torch.tensor
    ):
        head_d: torch.tensor = self.cnn.forwad(self.node_emb_d[head_index])
        tail_d: torch.tensor = self.cnn.forwad(self.node_emb_d[tail_index])

        head_s = self.node_emb_s(head_index)
        tail_s = self.node_emb_s(tail_index)

        rel_emb = self.rel_emb(rel_index)
        
        w_rel_emb = F.normalize(self.w_rel_emb(rel_index), p=self.p_norm, dim=-1)
        head_s = F.normalize(head_s, p=self.p_norm, dim=-1)
        tail_s = F.normalize(tail_s, p=self.p_norm, dim=-1)
        
        proj_head_d = head_d - (w_rel_emb * head_d).sum(dim=1).unsqueeze(dim=1) * w_rel_emb
        proj_tail_d = tail_d - (w_rel_emb * tail_d).sum(dim=1).unsqueeze(dim=1) * w_rel_emb
        
        proj_head_s = head_s - (w_rel_emb * head_s).sum(dim=1).unsqueeze(dim=1) * w_rel_emb
        proj_tail_s = tail_s - (w_rel_emb * tail_s).sum(dim=1).unsqueeze(dim=1) * w_rel_emb
        
        f_ss =  -((proj_head_s + rel_emb - proj_tail_s).norm(p=self.p_norm, dim=-1)) ** 2
        f_dd =  -((proj_head_d + rel_emb - proj_tail_d).norm(p=self.p_norm, dim=-1)) ** 2
        f_sd =  -((proj_head_s + rel_emb - proj_tail_d).norm(p=self.p_norm, dim=-1)) ** 2
        f_ds =  -((proj_head_d + rel_emb - proj_tail_s).norm(p=self.p_norm, dim=-1)) ** 2
        
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