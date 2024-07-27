import sys
sys.path.append("../")
import os
import torch
import pandas as pd
from typing import Literal
from typing import Tuple
from models.text_embedding.sentence_transformer import SentenceTransformer
from embedding import VoyageAI
from models.graph_embedding.transe import TransE
from models.graph_embedding.transh import TransH
from models.graph_embedding.rotate import RotatE
from torch_geometric.data import HeteroData
from utils.static_seed import static_seed

relation_dict = {
    "ParentOf": 0,
    "ChildOf": 1,
    "CanPrecede": 2, 
    "CanFollow": 3,
    "PeerOf":4 
}

static_seed(42)

root_path = os.path.join(os.path.dirname(__file__), "../")

class CreateHeteroData:
    def __init__(
        self,
        triplet: pd.DataFrame,
        dataset: pd.DataFrame,
        text_embedding_model: Literal["sbert", "voyage"],
        graph_embedding_model: Literal["transe", "transh", "rotate"]
    ):
        self.text_embedding_model = text_embedding_model
        self.graph_embedding_model = graph_embedding_model
        self.domain_mechanism = pd.read_csv(f"{root_path}/data/raw/capec_domain_mechanism.csv")
        self.dataset = dataset
        self.triplet = triplet
        self.relations = [
            "ParentOf", 
            "ChildOf", 
            "CanPrecede", 
            "CanFollow", 
            "PeerOf"
        ]
        
    
    @torch.no_grad()
    def _get_text_embedding(self, category: bool):
        embs = []
        if self.text_embedding_model == "sbert":
            model = SentenceTransformer()
            for series in self.dataset.to_dict(orient="records"):
                name = series["Name"] if isinstance(series["Name"], str) else ""
                description = series["Description"] if isinstance(series["Description"], str) else ""
                text = name + description
                emb = model.forward(
                    text_list=[text]
                )
                embs.append(emb)
        
        elif self.text_embedding_model == "voyage":
            model = VoyageAI()
            for series in self.dataset.to_dict(orient="records"):
                id = series["ID"]
                emb = model.read_embedding(id)
                
                if category:
                    names = self.domain_mechanism[self.domain_mechanism["ID"] == int(id[5:])]["Mechanism"].item().split(",")
                    category_emb = 0
                    for name in names:
                        category_emb += model.read_category_embedding(category=name.strip())
                
                    emb = torch.concat([emb, category_emb])   
                    # emb = emb + (category_emb/len(names)) 
                
                embs.append(emb)
        
        
        return torch.stack(embs)
    
    @torch.no_grad()
    def _get_graph_embedding(self):
        if self.graph_embedding_model == "transe":
            model = TransE(
                node_num=len(self.dataset),
                relation_num=len(self.relations),
                hidden_channels=256,
                margin=2.0,
                p_norm=2.0
            )
            model.load_state_dict(torch.load("./models/graph_embedding/kge.pth"))
            graph_embs = model.node_emb.weight + model.node_emb_im.weight
            return graph_embs
        
        if self.graph_embedding_model == "transh":
            model = TransH(
                node_num=len(self.dataset),
                relation_num=len(self.relations),
                hidden_channels=256,
                margin=2.0,
                p_norm=2.0                
            )
            model.load_state_dict(torch.load("./models/graph_embedding/kge.pth"))
            graph_embs = model.node_emb.weight
            return graph_embs
        
        if self.graph_embedding_model == "rotate":
            model = RotatE(
                node_num=len(self.dataset),
                relation_num=len(self.relations),
                hidden_channels=256,
                margin=2.0,
                p_norm=2.0
            )
            model.load_state_dict(torch.load("./models/graph_embedding/kge.pth"))
            graph_embs = model.node_emb.weight + model.node_emb_im.weight
            return graph_embs

        raise Exception("Not property KGE Model")
    
    def change_index(self):
        mapped_id = pd.DataFrame(
            data={
                "ID": self.dataset["ID"],
                "Name": self.dataset["Name"],
                "Description": self.dataset["Description"],
                "mappedID": range(len(self.dataset))
            }
        )

        mapped_relation = pd.DataFrame(
            data={
                "Relation": self.relations,
                "relationID": range(len(self.relations))
            }
        )

        triplets = []
        for triplet in self.triplet.to_dict(orient="records"):
            id1 = mapped_id[mapped_id["ID"] == triplet["ID1"]]["mappedID"].item()
            id2 = mapped_id[mapped_id["ID"] == triplet["ID2"]]["mappedID"].item()
            relation = mapped_relation[mapped_relation["Relation"] == triplet["Relation"]]["relationID"].item()

            triplets.append([id1, relation, id2])
        return torch.tensor(triplets)


    def reverse_triplet(self, triplets: torch.tensor):
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
        self,
        triplets: torch.Tensor,
        train: int=0.85,
        valid: int=0.05,
        test: int=0.1
    ) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:

        triplets = self.reverse_triplet(triplets=triplets)
        triple_num = len(triplets)
        static_seed(42)
        rnd_index = torch.randperm(triple_num)
        triplets = triplets[rnd_index]

        train_index = int(triple_num * train)
        valid_index = int(triple_num * valid)
        train_triples = triplets[:train_index]
        valid_triples = triplets[train_index:train_index+valid_index]
        test_triples = triplets[train_index+valid_index:]
        return train_triples, valid_triples, test_triples
    
    def make_edge(
            self, 
            relation: Literal["ParentOf", "ChildOf", "CanPrecede", "CanFollow", "PeerOf"],
            triplet: torch.tensor
        ):
        edge_label = relation_dict[relation]
        edge_index = torch.where(triplet[:, 1] == edge_label)[0]
        head_index = triplet[:, 0][edge_index]
        tail_index = triplet[:, 2][edge_index]
    
        return torch.stack([head_index, tail_index])
        
    
    def make_graph(
        self,
        embs: torch.tensor,
        triplet: torch.tensor
    ):
        hetero_data = HeteroData()
        
        hetero_data["capec"].x = embs
        hetero_data["capec"].node_id = range(len(self.dataset))
        hetero_data["capec", "to", "capec"].edge_label_index = torch.stack([triplet[:,0], triplet[:,2]])
        hetero_data["capec", "to", "capec"].edge_label = triplet[:,1]
        
        for relation in self.relations:
            edge_index = self.make_edge(
                relation=relation,
                triplet=triplet
            )
            hetero_data["capec", relation, "capec"].edge_label_index = edge_index
        
        return hetero_data
            
    
    def init_graph(self):
        self.text_embedding = self._get_text_embedding(category=True)
        self.graph_embedding = self._get_graph_embedding()
        triplet = self.change_index()
        train_triplet, valid_triplet, test_triplet = self.split_data(triplets=triplet)
        
        embs = torch.cat([self.graph_embedding, self.text_embedding], dim=1)
        # embs = self.text_embedding
        train_graph = self.make_graph(
            embs=embs,
            triplet=train_triplet
        )
        
        valid_graph = self.make_graph(
            embs=embs,
            triplet=valid_triplet
        )
        
        test_graph = self.make_graph(
            embs=embs,
            triplet=test_triplet
        )
        
        all_relations = ["to", "ParentOf", "ChildOf", "CanPrecede", "CanFollow", "PeerOf"]
        for relation in all_relations:
            train_graph["capec", relation, "capec"].edge_index = train_graph["capec", relation, "capec"].edge_label_index
            valid_graph["capec", relation, "capec"].edge_index = train_graph["capec", relation, "capec"].edge_label_index
            test_graph["capec", relation, "capec"].edge_index = torch.concat([train_graph["capec", relation, "capec"].edge_label_index, valid_graph["capec", relation, "capec"].edge_label_index], dim=1)

            
        
        return train_graph, valid_graph, test_graph
            
if __name__ == "__main__":
    dataset = pd.read_csv(f"{root_path}/data/processed/dataset.csv")[:559]
    triplet = pd.read_csv(f"{root_path}/data/processed/triplet.csv")[:722]
    
    create_data = CreateHeteroData(
        triplet=triplet,
        dataset=dataset,
        text_embedding_model="voyage",
        graph_embedding_model="rotate"
    )
    
    result = create_data.init_graph()
    
    print(result)
        
        
        
        
        
        
        
        
        
        