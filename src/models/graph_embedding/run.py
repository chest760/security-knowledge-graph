import os
import sys # noqa
sys.path.append("../../../") # noqa
import torch
from transe import transE
from transh import transH
from typing import Tuple
from src.utils.positive_triples import create_postive_triples, add_reverse_edge, dataset
from src.utils.triplet_loader import TripletLoader
import numpy as np
import random

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

"""
0: ParentOf
1: ChildOf
2: CanPrecede
3: CanFollow
4: PeerOf
5: TragetOf // CWE is target of CAPEC
6: AttackOf // CAPEC is attack of CWE
7: ExampleOf // CVE is example of CWE
"""

def split_data(
    triples: torch.Tensor,
    train: int=0.8,
    valid: int=0.1,
    test: int=0.1
) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    triple_num = len(triples)
    rnd_index = torch.randperm(triple_num)
    triples = triples[rnd_index]
    
    train_index = int(triple_num * train)
    valid_index = int(triple_num * valid)
    train_triples = triples[:train_index]
    valid_triples = triples[train_index:train_index+valid_index]
    test_triples = triples[train_index+valid_index:triple_num]
    
    train_triples = add_reverse_edge(train_triples)
    valid_triples = add_reverse_edge(valid_triples)
    test_triples = add_reverse_edge(test_triples)
    
    return train_triples, valid_triples, test_triples
    

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




device = 'cuda' if torch.cuda.is_available() else 'cpu'

model:transH = transH(
    node_num=len(dataset),
    relation_num=8,
    hidden_channels=256,
    margin=2.0
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

def triplet_loader(triplet: torch.tensor) -> TripletLoader:
    loader = TripletLoader(
        head_index=triplet[:, 0], 
        rel_type=triplet[:, 1],
        tail_index=triplet[:, 2],
        batch_size=32,
        shuffle=True,
    )
    
    return loader


def train(loader: TripletLoader):
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        neg_head_index, neg_rel_type, neg_tail_index = random_sample(head_index, rel_type, tail_index, len(dataset))
        loss = model.loss(
            head_index.to(device), 
            rel_type.to(device), 
            tail_index.to(device), 
            neg_head_index.to(device), 
            neg_rel_type.to(device), 
            neg_tail_index.to(device)
        )
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples

@torch.no_grad()
def valid(loader: TripletLoader):
    model.eval()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        neg_head_index, neg_rel_type, neg_tail_index = random_sample(head_index, rel_type, tail_index, len(dataset))
        loss = model.loss(
            head_index.to(device), 
            rel_type.to(device), 
            tail_index.to(device), 
            neg_head_index.to(device), 
            neg_rel_type.to(device), 
            neg_tail_index.to(device)
        )
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples


@torch.no_grad()
def test(
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
        batch_size: int,
        k: int = 10,
        log: bool = True,
    ) -> Tuple[float, float, float]:
        arange = range(head_index.numel())
        model.eval()

        mean_ranks, reciprocal_ranks, hits_at_k = [], [], []
        for i in arange:
            h, r, t = head_index[i], rel_type[i], tail_index[i]

            scores = []
            tail_indices = torch.arange(len(dataset), device=t.device)
            for ts in tail_indices.split(batch_size):
                scores.append(model.forward(h.expand_as(ts).to(device), r.expand_as(ts).to(device), ts.to(device)))
            rank = int((torch.cat(scores).argsort(
                descending=True) == t).nonzero().view(-1)) + 1
            mean_ranks.append(rank)
            reciprocal_ranks.append(1 / (rank))
            hits_at_k.append(rank <= k)

        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)

        return mean_rank, mrr, hits_at_k

def run(
    train_triplet: torch.tensor,
    valid_triplet: torch.tensor,
    test_triplet: torch.tensor
):  
    train_triplet.to(device)
    valid_triplet.to(device)
    test_triplet.to(device)
    
    train_loader = triplet_loader(train_triplet)
    valid_loader = triplet_loader(valid_triplet)
    for epoch in range(1, 101):
        loss = train(train_loader)
        valid_loss = valid(valid_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Valid_Loss: {valid_loss:.4f}')
        if epoch % 20 == 0:
            mean_rank, mrr, hits_at_k = test(
            head_index=test_triplet[:, 0],
            rel_type=test_triplet[:, 1],
            tail_index=test_triplet[:, 2],
            batch_size=64
            )
            print(f'MeanRank: {mean_rank}, MRR: {mrr}, Hits@10: {hits_at_k}')
        


if __name__ == "__main__":
    triples = create_postive_triples()  
    train_triples, valid_triples, test_triples = split_data(triples)
    
    run(
        train_triples,
        valid_triples,
        test_triples
    )
    
    
