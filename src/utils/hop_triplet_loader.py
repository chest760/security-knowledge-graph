from typing import List, Tuple, Dict

import torch
from torch import Tensor

class HopTripletLoader(torch.utils.data.DataLoader):
    def __init__(
        self, 
        triplet_dict: Dict[str, torch.tensor],
        index_key : str = "edge_index",
        label_key: str = "edge_label_index",
        neighbor_hop : int = 2,
        **kwargs
    ):
        
        self.edge_label_index_triplet: torch.tensor = triplet_dict[label_key]
        self.edge_index_triplets: torch.tensor = triplet_dict[index_key]
        self.node_id: torch.tensor = triplet_dict["node_id"]
        self.neighbor_hop = neighbor_hop

        super().__init__(
            range(self.edge_label_index_triplet[:, 0].numel()), 
            collate_fn=self.sample,
            **kwargs
        )
    
    def get_neighbor(
        self, 
        edge_label_index_triplets: torch.tensor
    ):
        neighbors = []
        def adjacent( 
            target_node: int
        ):
            head_adjacent = (self.edge_index_triplets[:, 0] == target_node).nonzero(as_tuple = True)[0]
            tail_adjacent = (self.edge_index_triplets[:, 2] == target_node).nonzero(as_tuple = True)[0]
            
            index = torch.cat([head_adjacent, tail_adjacent])
            
            adjacent_triplet = self.edge_index_triplets[index]
            
            return adjacent_triplet
        
        
             
        

    def sample(self, index: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        index = torch.tensor(index, device=self.edge_label_index_triplet.device)
        edge_label_index = self.edge_label_index_triplet[index]
        
        neighbors_triplets = self.get_neighbor(
            edge_label_index_triplets = edge_label_index
        )
        
        
        
        
        

        # return head_index, rel_type, tail_index
    


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

if __name__ == "__main__":
    import pandas as pd
    raw_triplet = pd.read_csv(f"../../data/processed/triplet.csv")
    raw_dataset = pd.read_csv(f"../../data/processed/dataset.csv")

    relations = [
        "ParentOf", "ChildOf", "CanPrecede", "CanFollow", "PeerOf", "TargetOf", "AttackOf", "InstanceOf", "AbstractionOf"
    ]
    mapped_id = pd.DataFrame(
        data={
            "ID": raw_dataset["ID"],
            "Name": raw_dataset["Name"],
            "Description": raw_dataset["Description"],
            "mappedID": range(len(raw_dataset))
        }
    )

    mapped_relation = pd.DataFrame(
        data={
            "Relation": relations,
            "relationID": range(len(relations))
        }
    )

    triplets = []
    for triplet in raw_triplet.to_dict(orient="records"):
        id1 = mapped_id[mapped_id["ID"] == triplet["ID1"]]["mappedID"].item()
        id2 = mapped_id[mapped_id["ID"] == triplet["ID2"]]["mappedID"].item()
        relation = mapped_relation[mapped_relation["Relation"] == triplet["Relation"]]["relationID"].item()
        triplets.append([id1, relation, id2])
    
    triplets = torch.tensor(triplets)
    
    train_triples, valid_triples, test_triples = split_data(triplets=triplets)
    
    
    loader = HopTripletLoader(
        triplet_dict=train_triples,
        neighbor_hop=2,
        batch_size=4
    )
    
    for triplet in loader:
        print(triplet)
