import torch
import torch.nn.functional as F

class GATLayer(torch.nn.Module):
    def __init__(
        self,
        in_dim: torch.Tensor,
        out_dim: torch.Tensor,
        dropout: torch.Tensor
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.attn_linear = torch.nn.Linear(2 * out_dim, 1, bias=False)
 
        
        self.leakyrelu = torch.nn.LeakyReLU()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.attn_linear.weight, gain=1.0)

    def forward(self, x, edge_index):
        
        # h = W*x
        h = self.linear(x)
        
        # バッチサイズごとにindexを振り直す
        edge_src, edge_dst = edge_index
        
        edge_src_h = h[edge_src]
        edge_dst_h = h[edge_dst]
        
        edge_features = torch.cat([edge_src_h, edge_dst_h], dim=-1)
        
        a = self.attn_linear(edge_features)
        e = F.leaky_relu(a)
        alpha = F.softmax(e, dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        
        h_prime = h.index_add(0, edge_src, alpha * edge_dst_h)
    
        
        return h_prime


class MultiHeadGATLayer(torch.nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        num_heads, 
        merge='cat'
    ):
        super().__init__()
        self.heads = torch.nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(
                GATLayer(
                    in_dim, 
                    out_dim,
                    dropout=0.2
                )
            )
        self.merge = merge
        
    def forward(self, x, edge_index):
        head_outs = [attn_head(x, edge_index) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

    def forward(self, h, edge_index):
        h = self.layer1(h, edge_index)
        h = F.elu(h)
        h = self.layer2(h, edge_index)
        return h


if __name__ == "__main__":
    x = torch.randn(100, 768)
    edge_index = torch.randint(0,99, (2, 50))
    
    gat = GAT(in_dim=768, hidden_dim=256, out_dim=256, num_heads=2)
    h = gat.forward(x, edge_index)
    
    print(h)
    print(h.size())
    