import sys # noqa
sys.path.append("../../") # noqa
import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple
from src.utils.positive_triples import create_postive_triples, add_reverse_edge
from src.utils.triplet_loader import TripletLoader
from src.utils.static_seed import static_seed
from model import Model
from word2vec import Word2vecModel
from make_data import create_tuple, dataset

static_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = os.path.join(os.path.dirname(__file__), "../../")

def split_data(
    triples: torch.Tensor,
    train: int=0.85,
    valid: int=0.05,
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
    
    # train_triples = add_reverse_edge(train_triples)
    # valid_triples = add_reverse_edge(valid_triples)
    # test_triples = add_reverse_edge(test_triples)
    
    return train_triples, valid_triples, test_triples

def triplet_loader(triplet: torch.tensor) -> TripletLoader:
    loader = TripletLoader(
        head_index=triplet[:, 0], 
        rel_type=triplet[:, 1],
        tail_index=triplet[:, 2],
        batch_size=64,
        shuffle=True,
    )
    
    return loader


def get_sentence_embedding():
    word2vec = Word2vecModel()
    
    embs = []
    sentences = dataset["Description"].tolist()
    for i, sentence in enumerate(sentences):
        if i % 200 == 0:
            print(i)
        if isinstance(sentence, float):
            sentence = ""
        emb = word2vec.sentence_to_word2vec(sentence)
        embs.append(emb)
        
    max_length = 150    

    return torch.tensor(embs)


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

text_embedding = get_sentence_embedding().to(device)
print(text_embedding.size())
text_embedding = text_embedding.mean(dim=2)

model = Model(
    node_num=len(dataset),
    rel_num=8,
    hidden_channels=100,
    pre_embedding=text_embedding,
    p_norm=2.0,
    margin=2.0
).to(device)

# triples = create_postive_triples()  
triples = create_tuple()
train_triples, valid_triples, test_triples = split_data(triples)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

train_loader = triplet_loader(train_triples)
valid_loader = triplet_loader(valid_triples)

def train():
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in train_loader:
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
def valid():
    model.eval()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in valid_loader:
        neg_head_index, neg_rel_type, neg_tail_index = random_sample(head_index, rel_type, tail_index, len(dataset))
        loss = model.loss(
            head_index.to(device), 
            rel_type.to(device), 
            tail_index.to(device), 
            neg_head_index.to(device), 
            neg_rel_type.to(device), 
            neg_tail_index.to(device),
            valid = False
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




for epoch in range(1, 501):
    loss = train()
    valid_loss = valid()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Valid_Loss: {valid_loss:.4f}')
    if epoch % 20 == 0:
        index = (test_triples[:, 1] == 1).nonzero(as_tuple=True)[0]
        mean_rank, mrr, hits_at_k = test(
            head_index=test_triples[:, 0][index],
            rel_type=test_triples[:, 1][index],
            tail_index=test_triples[:, 2][index],
            batch_size=64
        )
        print(f'MeanRank: {mean_rank}, MRR: {mrr}, Hits@10: {hits_at_k}')
    
    