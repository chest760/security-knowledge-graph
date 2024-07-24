import torch
from typing import Literal
from torch_geometric.nn import GATConv, HeteroConv
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.linear = torch.nn.Linear(512, 256)
        
        # out -> 512
        self.conv1 = GATConv(
            in_channels=(-1, -1),
            out_channels=128,
            heads=2,
            concat=True,
            add_self_loops=True,
        )

        # out -> 256
        self.conv2 = GATConv(
            in_channels=(-1, -1),
            out_channels=128,
            heads=2,
            concat=True,
            add_self_loops=True,
        )

        self.hetero_conv1 = HeteroConv({
            ('capec', 'ParentOf', 'capec'): self.conv1,
            ('capec', 'ChildOf', 'capec'): self.conv1,
            ('capec', 'CanPrecede', 'capec'): self.conv1,
            ('capec', 'CanFollow', 'capec'): self.conv1,
            ('capec', 'PeerOf', 'capec'): self.conv1,
        }, aggr='sum')
        
        self.hetero_conv2 = HeteroConv({
            ('capec', 'ParentOf', 'capec'): self.conv2,
            ('capec', 'ChildOf', 'capec'): self.conv2,
            ('capec', 'CanPrecede', 'capec'): self.conv2,
            ('capec', 'CanFollow', 'capec'): self.conv2,
            ('capec', 'PeerOf', 'capec'): self.conv2,
        }, aggr='sum')

    def forward(
        self,
        x_dict: torch.tensor,
        edge_index_dict: torch.tensor
    ):

        
        x_dict = self.hetero_conv1(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
        )
        
        # x_dict = self.hetero_conv2(
        #     x_dict=x_dict,
        #     edge_index_dict=edge_index_dict,
        # )
        
        # x_dict = { node_type: F.relu(x) for node_type, x in x_dict.items() }
        # x_dict = { node_type: self.linear(x) for node_type, x in x_dict.items() }
        
        return x_dict
