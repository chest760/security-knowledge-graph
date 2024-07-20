from typing import List, Tuple, Dict

import torch
from torch import Tensor
from src.utils.static_seed import static_seed

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


class HopTripletLoader(torch.utils.data.DataLoader):
    def __init__(
        self, 
        triplet_dict: Dict[str, torch.tensor],
        index_key : str = "edge_index",
        label_key: str = "edge_label_index",
        neighbor_hop : int = 1,
        add_negative_label: bool = True,
        num_node: int = None,
        **kwargs
    ):
        
        self.edge_label_index_triplet: torch.tensor = triplet_dict[label_key]
        self.edge_index_triplets: torch.tensor = triplet_dict[index_key]
        self.node_id: torch.tensor = triplet_dict["node_id"]
        self.neighbor_hop = neighbor_hop
        
        self.index_key = index_key
        self.label_key = label_key
        self.add_negative_label = add_negative_label
        self.num_node = num_node

        super().__init__(
            range(self.edge_label_index_triplet[:, 0].numel()), 
            collate_fn=self.sample,
            **kwargs
        )
    
    def _new_node_id(
        self, 
        neighbor_triplet: torch.tensor,
        label_triplet: torch.tensor
    ):
        all_nodes = torch.concat([neighbor_triplet[:, 0], neighbor_triplet[:, 2], label_triplet[:, 0], label_triplet[:, 2]]).unique()
        
        sorted_nodes = all_nodes.sort()
        
        original_node_index = sorted_nodes.values
        new_node_index = sorted_nodes.indices
        
        neighbor_head_index = torch.where(original_node_index == neighbor_triplet[:, 0].unsqueeze(dim=1))[1]
        neighbor_tail_index = torch.where(original_node_index == neighbor_triplet[:, 2].unsqueeze(dim=1))[1]

        label_head_index = torch.where(original_node_index == label_triplet[:, 0].unsqueeze(dim=1))[1]
        label_tail_index = torch.where(original_node_index == label_triplet[:, 2].unsqueeze(dim=1))[1]
        
        neighbor_triplet[:, 0] = neighbor_head_index
        neighbor_triplet[:, 2] = neighbor_tail_index

        label_triplet[:, 0] = label_head_index
        label_triplet[:, 2] = label_tail_index
        
        if self.add_negative_label:
            result = {
                "node_id": original_node_index,
                self.index_key: neighbor_triplet,
                self.label_key: label_triplet,
                "positive": label_triplet[:int(len(label_triplet)/2)]    ,
                "negative": label_triplet[int(len(label_triplet)/2):]        
            }
        else:
            result = {
                "node_id": original_node_index,
                self.index_key: neighbor_triplet,
                self.label_key: label_triplet   
            }
        
        return result
        
        
    
    def _get_neighbor(
        self, 
        edge_label_index_triplets: torch.tensor
    ):
        neighbors = []
        def adjacent( 
            head_index: torch.tensor,
            tail_index: torch.tensor
        ):
            # [h, r, t]のhに隣接するノードを取得
            head_adjacent_index_head = torch.isin(self.edge_index_triplets[:, 0], head_index).nonzero(as_tuple = True)[0]
            head_adjacent_index_tail = torch.isin(self.edge_index_triplets[:, 2], head_index).nonzero(as_tuple = True)[0]
            
            # [h, r, t]のtに隣接するノードを取得
            tail_adjacent_index_head = torch.isin(self.edge_index_triplets[:, 0], tail_index).nonzero(as_tuple = True)[0]
            tail_adjacent_index_tail = torch.isin(self.edge_index_triplets[:, 2], tail_index).nonzero(as_tuple = True)[0]
            
            head_adjacent_index = torch.cat([head_adjacent_index_head, head_adjacent_index_tail])
            tail_adjacent_index = torch.cat([tail_adjacent_index_head, tail_adjacent_index_tail])
            
            adjacent_index = torch.cat([head_adjacent_index, tail_adjacent_index])          
            adjacent_triplet = self.edge_index_triplets[adjacent_index]
            
            return adjacent_triplet
        
        for i in range(self.neighbor_hop):
            if i == 0:
                head_index: torch.tensor = edge_label_index_triplets[:, 0]
                tail_index: torch.tensor = edge_label_index_triplets[:, 2]
            else:
                head_index: torch.tensor = adjacent_triplet[:, 0]
                tail_index: torch.tensor = adjacent_triplet[:, 2]
            
            adjacent_triplet = adjacent(
                head_index=head_index,
                tail_index=tail_index
            )
            
            neighbors.append(adjacent_triplet)
        
        neighbors = torch.cat(neighbors, dim=0).view(-1, 3).unique(dim=0)
        
        return neighbors
                
             
    def sample(self, index: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        index = torch.tensor(index, device=self.edge_label_index_triplet.device)
        edge_label_index = self.edge_label_index_triplet[index]
                
        if self.add_negative_label:
            neg_head_index, neg_rel_type, neg_tail_index = random_sample(
                head_index=edge_label_index[:, 0],
                rel_type=edge_label_index[:, 1],
                tail_index=edge_label_index[:, 2],
                num_node=self.num_node
                
            )
            negative_triplet = torch.stack([neg_head_index, neg_rel_type, neg_tail_index]).transpose(0, 1)
            
            edge_label_index = torch.cat([edge_label_index, negative_triplet])
        
        neighbors_triplets = self._get_neighbor(
            edge_label_index_triplets = edge_label_index
        )
        
        result = self._new_node_id(
            neighbor_triplet=neighbors_triplets,
            label_triplet=edge_label_index
        )
        
        
        return result
        
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
        batch_size=2
    )
    
    for triplet in loader:
        print(triplet)
