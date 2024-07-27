import torch
from torch_geometric.nn import RGATConv
import torch.nn.functional as F

class RGAT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        heads: int
    ):
        super().__init__()
        
        self.conv1 =  RGATConv(
            in_channels, 
            256, 
            num_relations,
            heads=heads
        )
        
        self.conv2 = RGATConv(
            512, 
            out_channels, 
            num_relations,
            heads=heads
        )
    
    
    def forward(
        self,
        x_dict: torch.tensor,
        edge_index: torch.tensor,
        edge_label: torch.tensor
    ):
        x = x_dict["capec"]

        x = self.conv1(
            x=x,
            edge_index=edge_index,
            edge_type=edge_label
        )
        # x =  F.relu(x) 
        # x = self.conv2(
        #     x=x,
        #     edge_index=edge_index,
        #     edge_type=edge_label
        # )
        
        x_dict = {"capec": x}
        
        return x_dict
        