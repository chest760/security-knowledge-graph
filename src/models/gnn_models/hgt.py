import torch
from torch_geometric.nn import HGTConv
import torch.nn.functional as F

class HGT(torch.nn.Module):
    def __init__(
        self, 
        hidden_channels, 
        out_channels, 
        data
    ):
        super().__init__()
        self.conv1 = HGTConv(
                        hidden_channels, 
                        256, 
                        data.metadata(),
                        heads=2
                    )

        self.conv2 = HGTConv(
                        512, 
                        out_channels, 
                        data.metadata(),
                    )
        
        self.conv1.reset_parameters()
        
        self.linear = torch.nn.Linear(256 + 1024*2, 256)
    
    def forward(
        self,
        x_dict: torch.tensor,
        edge_index_dict: torch.tensor,
    ):
        x_dict = self.conv1(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
        )
        
        # x_dict = {node_type: F.relu(x) for node_type, x in x_dict.items() }

        # x_dict = self.conv2(
        #     x_dict=x_dict,
        #     edge_index_dict=edge_index_dict,
        # )
        
        # x_dict = { node_type: self.linear(x) for node_type, x in x_dict.items() }
        
        return x_dict