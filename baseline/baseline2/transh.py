import math
import torch
from typing import Tuple
import torch.nn.functional as F
from src.utils.triplet_loader import TripletLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TransH(torch.nn.Module):
    def __init__(
        self,
        node_num: int,
        rel_num: int,
        hidden_channels: int,
        p_norm: float,
        margin: float
        
    ):
        super().__init__()
        
        self.node_emb = torch.nn.Embedding(node_num, hidden_channels)
        self.rel_emb = torch.nn.Embedding(rel_num, hidden_channels)
        self.w_rel_emb = torch.nn.Embedding(rel_num, hidden_channels)
        self.p_norm = p_norm
        self.margin = margin
        self.hidden_channels = hidden_channels
        
    def reset_prameters(self):
        bound = 6. / math.sqrt(self.hidden_channels)
        torch.nn.init.uniform_(self.node_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        torch.nn.init.uniform_(self.w_rel_emb.weight, -bound, bound)
        F.normalize(self.rel_emb.weight.data, p=self.p_norm, dim=-1, out=self.rel_emb.weight.data)
        F.normalize(self.w_rel_emb.weight.data, p=self.p_norm, dim=-1, out=self.w_rel_emb.weight.data)
    
    def forward(
        self,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor
    ) -> torch.tensor:
        head_emb = self.node_emb(head_index)
        tail_emb = self.node_emb(tail_index)
        rel_emb = self.rel_emb(rel_type)
        
        w_rel_rmb = F.normalize(self.w_rel_emb(rel_type), p=self.p_norm, dim=-1)
        head_emb = F.normalize(head_emb, p=self.p_norm, dim=-1)
        tail_emb = F.normalize(tail_emb, p=self.p_norm, dim=-1)
        
        h_proj = head_emb - (w_rel_rmb * head_emb).sum(dim=1).unsqueeze(dim=1) * w_rel_rmb
        t_proj = tail_emb - (w_rel_rmb * tail_emb).sum(dim=1).unsqueeze(dim=1) * w_rel_rmb
        

        score = -((h_proj + rel_emb - t_proj ).norm(p=self.p_norm, dim=-1))**2
        
        return score
    
    def loss(
        self,
        head_index: torch.tensor,
        rel_type: torch.tensor,
        tail_index: torch.tensor,
        neg_head_index: torch.tensor,
        neg_rel_type: torch.tensor,
        neg_tail_index: torch.tensor
    ) -> torch.tensor:
        
        pos_score = self.forward(
            head_index=head_index,
            rel_type=rel_type,
            tail_index=tail_index
        )
        
        neg_score = self.forward(
            head_index=neg_head_index,
            rel_type=neg_rel_type,
            tail_index=neg_tail_index
        )
        
        loss = F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )
        
        return loss


def triplet_loader(triplet: torch.tensor) -> TripletLoader:
    loader = TripletLoader(
        head_index=triplet[:, 0], 
        rel_type=triplet[:, 1],
        tail_index=triplet[:, 2],
        batch_size=64,
        shuffle=True,
    )
    
    return loader


@torch.no_grad()
def random_sample(
    head_index: torch.Tensor, 
    rel_type: torch.Tensor,
    tail_index: torch.Tensor,
    num_node: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_negatives = head_index.numel() // 2
    rnd_index = torch.randint(num_node, head_index.size(),device=head_index.device)
    
    head_index = head_index.clone()
    head_index[:num_negatives] = rnd_index[:num_negatives]
    tail_index = tail_index.clone()
    tail_index[num_negatives:] = rnd_index[num_negatives:]

    return head_index, rel_type, tail_index


class TransHTrain:
    def __init__(
        self,
        node_num: int,
        rel_num: int,
        hidden_channels: int,
        p_norm: float,
        margin: float
    ) -> None:
        self.model = TransH(
            node_num=node_num,
            rel_num=rel_num,
            hidden_channels=hidden_channels,
            p_norm=p_norm,
            margin=margin
        )
        
        self.node_num = node_num
        self.rel_num = rel_num
        self.hidden_channels = hidden_channels
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
    
    
    def run(
        self,
        train_triplet: torch.tensor,
        valid_triplet: torch.tensor,
        test_triplet: torch.tensor
    ):
        train_loader = triplet_loader(train_triplet)
        valid_loader = triplet_loader(valid_triplet)
        test_loader = triplet_loader(test_triplet)
        
        for epoch in range(1, 301):
            train_loss = self.train()
            valid_loss = self.valid()
            print(f"Train Loss: {train_loss}, Valid Loss: {valid_loss}")
            if epoch % 20 == 0:
                self.test()    

    
    def train(
        self,
        loader
    ):
        self.model.train()
        total_loss = total_examples = 0
        for head_index, rel_type, tail_index in loader:
            self.optimizer.zero_grad()
            neg_head_index, neg_rel_type, neg_tail_index = random_sample(head_index, rel_type, tail_index, self.node_num)
            loss = self.model.loss(
                head_index.to(device), 
                rel_type.to(device), 
                tail_index.to(device), 
                neg_head_index.to(device), 
                neg_rel_type.to(device), 
                neg_tail_index.to(device)
            )
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * head_index.numel()
            total_examples += head_index.numel()
        return total_loss / total_examples
    
    @torch.no_grad()
    def valid(
        self,
        loader
    ):
        self.model.eval()
        total_loss = total_examples = 0
        for head_index, rel_type, tail_index in loader:
            neg_head_index, neg_rel_type, neg_tail_index = random_sample(head_index, rel_type, tail_index, self.node_num)
            loss = self.model.loss(
                head_index.to(device), 
                rel_type.to(device), 
                tail_index.to(device), 
                neg_head_index.to(device), 
                neg_rel_type.to(device), 
                neg_tail_index.to(device),
            )
            total_loss += float(loss) * head_index.numel()
            total_examples += head_index.numel()
        return total_loss / total_examples
    
    @torch.no_grad()
    def test(
        self,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
        train_triples,
        valid_triples,
        batch_size: int,
        k: int = 10,
    ):
        arange = range(head_index.numel())
        self.model.eval()

        exist_data = torch.cat([train_triples, valid_triples])
        mean_ranks, reciprocal_ranks, hits_at_k = [], [], []

        for i in arange:
            h, r, t = head_index[i], rel_type[i], tail_index[i]

            exist = exist_data[exist_data[:,0] == h.item()]
            exist_tail = exist[exist[:,1] == r.item()][:, 2]
            scores = []
            tail_indices = torch.arange(self.node_num)
            tail_indices = torch.tensor(list(set(tail_indices.tolist()) - set(exist_tail.tolist())))
            t = (tail_indices == t).nonzero(as_tuple=True)[0].to(device)

            for ts in tail_indices.split(batch_size):
                scores.append(self.model.forward(h.expand_as(ts).to(device), r.expand_as(ts).to(device), ts.to(device)))
            rank = int((torch.cat(scores).argsort(descending=True) == t).nonzero().view(-1)) + 1
            mean_ranks.append(rank)
            reciprocal_ranks.append(1 / (rank))
            hits_at_k.append(rank <= k)
        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)
        return mean_rank, mrr, hits_at_k
        
    def save(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)
        pass
        
         