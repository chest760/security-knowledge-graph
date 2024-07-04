import sys # noqa
sys.path.append("../../") # noqa
import os
import torch
import pandas as pd
from typing import Tuple
from src.utils.triplet_loader import TripletLoader
from src.utils.static_seed import static_seed

root_path = os.path.join(os.path.dirname(__file__), "../../")


def change_index(data:pd.DataFrame, triplets_df: pd.DataFrame):
    relations = [
        "ParentOf", "ChildOf", "CanPrecede", "CanFollow", "PeerOf", "TargetOf", "AttackOf", "ExampleOf"
    ]
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
    
    train_triples = reverse_triplet(train_triples)
    valid_triples = reverse_triplet(valid_triples)
    test_triples = reverse_triplet(test_triples)
    
    return train_triples, valid_triples, test_triples

def triplet_loader(
    triplet: torch.tensor) -> TripletLoader:
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
    num_node: int,
    seed :int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_negatives = head_index.numel() // 2
    static_seed(seed)
    rnd_index = torch.randint(num_node, head_index.size(),device=head_index.device)
    head_index = head_index.clone()
    head_index[:num_negatives] = rnd_index[:num_negatives]
    tail_index = tail_index.clone()
    tail_index[num_negatives:] = rnd_index[num_negatives:]

    return head_index, rel_type, tail_index




def run(
    train_triplet: torch.tensor,
    valid_triplet: torch.tensor,
    test_triplet: torch.tensor
):
    pass





if __name__ == "__main__":
    data = pd.read_csv(f"{root_path}/data/processed/dataset.csv")
    triplets_df = pd.read_csv(f"{root_path}/data/processed/triplet.csv")
    triplet = change_index(data=data, triplets_df=triplets_df)
    train, valid, test = split_data(triples=triplet)
    
