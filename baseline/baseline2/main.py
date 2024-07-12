import sys # noqa
sys.path.append("../../") # noqa
import os
import torch
import pandas as pd
from typing import Tuple
from src.utils.triplet_loader import TripletLoader
from model import BaseLineModel2
from src.utils.static_seed import static_seed
from sentence_bert import SentenceBert

root_path = os.path.join(os.path.dirname(__file__), "../../")
static_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
relations = [
        "ParentOf", 
        "ChildOf", 
        "CanPrecede", 
        "CanFollow", 
        "PeerOf", 
        "TargetOf", 
        "AttackOf", 
        "InstanceOf", 
        "AbstractionOf"
    ]

raw_triplet = pd.read_csv(f"{root_path}/data/processed/triplet.csv")
raw_dataset = pd.read_csv(f"{root_path}/data/processed/dataset.csv")

node_num = len(raw_dataset)
rel_num = len(relations)

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
        if relation == 0 or relation == 2 or relation == 5:
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
    train_triplets_dict = {}
    valid_triplets_dict = {}
    test_triplets_dict = {}
    
    triplets = reverse_triplet(triplets=triplets)
    triple_num = len(triplets)
    rnd_index = torch.randperm(triple_num)
    triplets = triplets[rnd_index]
    
    train_index = int(triple_num * train)
    valid_index = int(triple_num * valid)
    train_triples = triplets[:train_index]
    valid_triples = triplets[train_index:train_index+valid_index]
    test_triples = triplets[train_index+valid_index:]
    
    train_triplets_dict["node_id"] = torch.arange(len(raw_dataset))
    valid_triplets_dict["node_id"] = torch.arange(len(raw_dataset))
    test_triplets_dict["node_id"] = torch.arange(len(raw_dataset))
    
    train_triplets_dict["edge_label_index"] = train_triples
    valid_triplets_dict["edge_label_index"] = valid_triples
    test_triplets_dict["edge_label_index"] = test_triples
    
    train_triplets_dict["edge_index"] = train_triples
    valid_triplets_dict["edge_index"] = train_triples
    test_triplets_dict["edge_index"] = torch.concat([train_triples, valid_triples])
    
    return train_triplets_dict, valid_triplets_dict, test_triplets_dict

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
    model: BaseLineModel2,
    loader: TripletLoader
):
    model.train()


@torch.no_grad()
def valid(
    model: BaseLineModel2,
    loader: TripletLoader
):
    model.eval()


@torch.no_grad()
def test(
    model: BaseLineModel2,
    loader: TripletLoader
):
    model.eval()

def main():    
    triplet = change_index(data=raw_dataset, triplets_df=raw_triplet)
    train_triples, valid_triples, test_triples = split_data(triplets=triplet)
    print("######################")
    print(train_triples)
    
    train_loader = triplet_loader(triplet=train_triples)
    valid_loader = triplet_loader(triplet=valid_triples)
    
        
    
    sentence_bert = SentenceBert()
    
    embs = []
    for series in raw_dataset.to_dict(orient="records"):
        name = series["Name"] if isinstance(series["Name"], str) else ""
        desciption = series["Description"] if isinstance(series["Description"], str) else ""
        sentence = name + " " + desciption
        emb = sentence_bert(sentence=sentence)
        embs.append(emb)
        
        
    text_embedding = torch.cat(embs, dim=0)
    model = BaseLineModel2(
        structure_node_embedding=torch.randn(559, 128),
        structure_rel_embedding=torch.randn(5, 128),
        text_node_embedding=embs
        
    )

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