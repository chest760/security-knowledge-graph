import sys # noqa
sys.path.append("../../") # noqa
import os
import torch
import pandas as pd
from typing import Tuple
from src.utils.triplet_loader import TripletLoader
from model import BaseLineModel1
from src.utils.static_seed import static_seed
from word2vec import Word2vecModel

root_path = os.path.join(os.path.dirname(__file__), "../../")
static_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
relations = [
        "ParentOf", "ChildOf", "CanPrecede", "CanFollow", "PeerOf", "TargetOf", "AttackOf", "InstanceOf", "AbstractionOf"
    ]

raw_triplet = pd.read_csv(f"{root_path}/data/processed/triplet.csv")
raw_dataset = pd.read_csv(f"{root_path}/data/processed/dataset.csv")

def change_index(data:pd.DataFrame, triplets_df: pd.DataFrame):
    mapped_id = pd.DataFrame(
        data={
            "ID": data["ID"],
            "Name": data["Name"],
            "Description": data["Description"],
            "mappedID": range(len(data))
        }
    )
    
    mapped_relation = pd.DataFrame(
        data={
            "Relation": relations,
            "relationID": range(len(relations))
        }
    )
    
    triplets = []
    for triplet in triplets_df.to_dict(orient="records"):
        id1 = mapped_id[mapped_id["ID"] == triplet["ID1"]]["mappedID"].item()
        id2 = mapped_id[mapped_id["ID"] == triplet["ID2"]]["mappedID"].item()
        relation = mapped_relation[mapped_relation["Relation"] == triplet["Relation"]]["relationID"].item()
        
        triplets.append([id1, relation, id2])
    return torch.tensor(triplets)

def reverse_triplet(triplets: torch.tensor):
    new_triplets = []
    for triplet in triplets:
        id1 = triplet[0].item()
        relation = triplet[1].item()
        id2 = triplet[2].item()
        new_triplets.append([id1, relation, id2])
        if relation == 0 or relation == 2 or relation == 5 or relation == 7:
            new_triplets.append([id2, relation+1, id1])
        elif relation == 4:
            new_triplets.append([id2, relation, id1])
            
    return torch.tensor(new_triplets)

def split_data(
    triplets: torch.Tensor,
    train: int=0.85,
    valid: int=0.05,
    test: int=0.1
) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    triplets = reverse_triplet(triplets=triplets)
    triple_num = len(triplets)
    rnd_index = torch.randperm(triple_num)
    triplets = triplets[rnd_index]
    
    train_index = int(triple_num * train)
    valid_index = int(triple_num * valid)
    train_triples = triplets[:train_index]
    valid_triples = triplets[train_index:train_index+valid_index]
    test_triples = triplets[train_index+valid_index:]
    
    # train_triples = reverse_triplet(train_triples)
    # valid_triples = reverse_triplet(valid_triples)
    # test_triples = reverse_triplet(test_triples)
    
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

def train(
    loader:TripletLoader,
    model: BaseLineModel1, 
    optimizer: any
    ):
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        neg_head_index, neg_rel_type, neg_tail_index = random_sample(head_index, rel_type, tail_index, len(raw_dataset))
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
def valid(
    loader:TripletLoader,
    model: BaseLineModel1
    ):
    model.eval()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        neg_head_index, neg_rel_type, neg_tail_index = random_sample(head_index, rel_type, tail_index, len(raw_dataset))
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
    model: BaseLineModel1,
    head_index: torch.Tensor,
    rel_type: torch.Tensor,
    tail_index: torch.Tensor,
    train_triples,
    valid_triples,
    batch_size: int,
    k: int = 10,
):
    arange = range(head_index.numel())
    model.eval()
    
    exist_data = torch.cat([train_triples, valid_triples])
    mean_ranks, reciprocal_ranks, hits_at_k = [], [], []
    
    for i in arange:
        h, r, t = head_index[i], rel_type[i], tail_index[i]
        
        exist = exist_data[exist_data[:,0] == h.item()]
        exist_tail = exist[exist[:,1] == r.item()][:, 2]
        scores = []
        tail_indices = torch.arange(len(raw_dataset))
        tail_indices = torch.tensor(list(set(tail_indices.tolist()) - set(exist_tail.tolist())))
        
        t = (tail_indices == t).nonzero(as_tuple=True)[0].to(device)
    
        
        for ts in tail_indices.split(batch_size):
            scores.append(model.forward(h.expand_as(ts).to(device), r.expand_as(ts).to(device), ts.to(device)))
        rank = int((torch.cat(scores).argsort(descending=True) == t).nonzero().view(-1)) + 1
        mean_ranks.append(rank)
        reciprocal_ranks.append(1 / (rank))
        hits_at_k.append(rank <= k)
    mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
    mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
    hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)
    return mean_rank, mrr, hits_at_k


def main():    
    triplet = change_index(data=raw_dataset, triplets_df=raw_triplet)
    train_triples, valid_triples, test_triples = split_data(triplets=triplet)
    
    train_loader = triplet_loader(triplet=train_triples)
    valid_loader = triplet_loader(triplet=valid_triples)
    
    word2vec = Word2vecModel()
    
    embs = []
    for series in raw_dataset.to_dict(orient="records"):
        name = series["Name"] if isinstance(series["Name"], str) else ""
        desciption = series["Description"] if isinstance(series["Description"], str) else ""
        sentence = name + " " + desciption
        emb = word2vec.sentence_to_word2vec(sentence=sentence)
        embs.append(emb)
    
    text_embedding = torch.tensor(embs).to(device)
    
    

    model = BaseLineModel1(
        node_num=len(raw_dataset),
        rel_num=len(relations),
        node_dim=100,
        text_embedding=text_embedding,
        text_embedding_dim=300,
        margin=2.0,
        p_norm=2.0
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(1, 501):
        train_loss = train(
            loader=train_loader, 
            model=model,
            optimizer=optimizer
        )
        valid_loss = valid(
            loader=valid_loader,
            model=model
        )
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Valid_Loss: {valid_loss:.4f}')
        
        if epoch % 30 == 0:
            index = (test_triples[:, 1] == 1 ).nonzero(as_tuple=True)[0]
            mean_rank, mrr, hits_at_k = test(
                model=model,
                head_index=test_triples[:, 0],
                rel_type=test_triples[:, 1],
                tail_index=test_triples[:, 2],
                train_triples=train_triples,
                valid_triples=valid_triples,
                batch_size=64
            )
            print(f'MeanRank: {mean_rank:.4f}, MRR: {mrr:.4f}, Hits@10: {hits_at_k:.4f}')
    
    
main()