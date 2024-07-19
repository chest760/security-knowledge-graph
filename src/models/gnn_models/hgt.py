import torch
from torch_geometric.nn import HGTConv

class HGT(torch.nn.Module):
    def __init__(
        self, 
        hidden_channels, 
        out_channels, 
        num_heads, 
        num_layers,
        data
    ):
        super().__init__()
        self.conv1 = HGTConv(
                        hidden_channels, 
                        hidden_channels, 
                        data.metadata(),
                        num_heads
                    )
        
        self.linear = torch.nn.Linear(512, 128)
    
    def forward(
        self,
        x_dict: torch.tensor,
        edge_index_dict: torch.tensor,
    ):
        x_dict = self.conv1(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
        )
        x_dict = { node_type: self.linear(x) for node_type, x in x_dict.items() }
        
        return x_dict