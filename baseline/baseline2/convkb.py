import torch

class ConvKB(torch.nn.Module):
    def __init__(
        self,
        node_num: int,
        rel_num: int,
        kernel_size: int,
        hidden_channels: int,
        out_channels: int
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        self.ent_embeddings = torch.nn.Embedding(node_num, hidden_channels) 
        self.rel_embeddings = torch.nn.Embedding(rel_num, hidden_channels)

        self.conv1_bn = torch.nn.BatchNorm1d(3)
        self.conv_layer = torch.nn.Conv1d(3, out_channels, kernel_size)
        self.conv2_bn = torch.nn.BatchNorm1d(out_channels)
        self.dropout = torch.nn.Dropout(0.2)
        self.non_linearity = torch.nn.ReLU()
        self.fc_layer = torch.nn.Linear((hidden_channels - kernel_size + 1) * out_channels, 1, bias=False)
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc_layer.weight.data)
        torch.nn.init.xavier_uniform_(self.conv_layer.weight.data)
    
    
    def _calc_score(
        self,
        h: torch.tensor,
        r: torch.tensor,
        t: torch.tensor
    ):
        h = h.unsqueeze(1)
        r = r.unsqueeze(1)
        t = t.unsqueeze(1)
    
        conv_input = torch.cat([h, r, t], 1) 
        # conv_input = conv_input.transpose(1, 2)
        # conv_input = conv_input.unsqueeze(1)
        conv_input = self.conv1_bn(conv_input)
        out_conv = self.conv_layer(conv_input)
        out_conv = self.conv2_bn(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, (self.hidden_channels - self.kernel_size + 1) * self.out_channels)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc).view(-1)

        return score
    
    
    def forward(
        self,
        x: torch.tensor,
        rel_emb: torch.tensor,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor
    ) -> torch.tensor:
        h = x[head_index]
        r = rel_emb(rel_type)
        t = x[tail_index]
        score = self._calc_score(h, r, t)
        
        # regularization
        l2_reg = 0
        for W in self.conv_layer.parameters():
            l2_reg = l2_reg + W.norm(p=2.0) ** 2
        for W in self.fc_layer.parameters():
            l2_reg = l2_reg + W.norm(p=2.0) ** 2
        
        return score, l2_reg