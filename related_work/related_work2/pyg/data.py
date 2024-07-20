import sys
sys.path.append("../")
import os
import torch
import numpy as np
import pandas as pd
import torch_geometric.transforms as T
# from word2vec import Word2vecModel
from torch_geometric.data import HeteroData
root = os.path.join(os.path.dirname(__file__), "../../../")

dataset = pd.read_csv(f"{root}/data/processed/dataset.csv")
triplets_df = pd.read_csv(f"{root}/data/processed/triplet.csv")
rev_triplets_df = pd.read_csv(f"{root}/data/processed/reverse_triplet.csv")

triplets_df = pd.concat([triplets_df, rev_triplets_df])

map_relation = {
    "ParentOf": 0, "ChildOf": 1, "CanPrecede": 2, "CanFollow": 3, "PeerOf": 4, "TargetOf":5,  "AttackOf": 6, "ExampleOf": 7
}


class CreateData:
    def __init__(self) -> None:
        # self.word2vec = Word2vecModel()
        self.capec = dataset[dataset["ID"].str.contains("capec")].reset_index(drop=True)
        self.cwe = dataset[dataset["ID"].str.contains("cwe")].reset_index(drop=True)
        self.cve = dataset[dataset["ID"].str.contains("CVE")].reset_index(drop=True)
        
        self.capec_mapped_df = pd.DataFrame(data={"ID": self.capec["ID"], "mappedID": torch.arange(len(self.capec))})
        self.cwe_mapped_df = pd.DataFrame(data={"ID": self.cwe["ID"], "mappedID": torch.arange(len(self.cwe))})
        self.cve_mapped_df = pd.DataFrame(data={"ID": self.cve["ID"], "mappedID": torch.arange(len(self.cve))})
        
        self.create_base_data()
        self.create_edge()
    
    def get_embedding(self, sentences: list[str]):
        embs = []
        for sentence in sentences:
            emb = self.word2vec.sentence_to_word2vec(sentence)
            embs.append(emb)
        
        return torch.tensor(embs)
    
        
    def create_base_data(self):
        self.data = HeteroData()
        self.data["capec"].node_id = torch.arange(len(self.capec))
        self.data["cwe"].node_id = torch.arange(len(self.cwe))
        self.data["cve"].node_id = torch.arange(len(self.cve))
        self.data["capec"].x = torch.randn(len(self.capec), 768)
        self.data["cwe"].x = torch.randn(len(self.cwe), 768)
        self.data["cve"].x = torch.randn(len(self.cve), 768)
        
    def get_mapped_id(self, id: str):
        if "capec" in id:
            return self.capec_mapped_df[self.capec_mapped_df["ID"] == id]["mappedID"].item()
        if "cwe" in id:
            return self.cwe_mapped_df[self.cwe_mapped_df["ID"] == id]["mappedID"].item()
        if "CVE" in id:
            return self.cve_mapped_df[self.cve_mapped_df["ID"] == id]["mappedID"].item()
    
    def create_edge(self):
        capec_to_capec = []
        cwe_to_cwe = []
        cwe_to_capec = []
        cve_to_cwe = []
        for series in triplets_df.to_dict(orient="records"):
            id1 = series["ID1"]
            id2 = series["ID2"]
            rel = series["Relation"]
            if "capec" in id1 and "capec" in id2:
                capec_to_capec.append([self.get_mapped_id(id1), map_relation[rel], self.get_mapped_id(id2)])
            elif "cwe" in id1 and "cwe" in id2:
                cwe_to_cwe.append([self.get_mapped_id(id1), map_relation[rel], self.get_mapped_id(id2)])
            elif "cwe" in id1 and "capec" in id2:
                cwe_to_capec.append([self.get_mapped_id(id1), map_relation[rel], self.get_mapped_id(id2)])
            elif "CVE" in id1 and "cwe" in id2:
                cve_to_cwe.append([self.get_mapped_id(id1), map_relation[rel], self.get_mapped_id(id2)])
            
        
        capec_to_capec = torch.tensor(capec_to_capec)
        cwe_to_cwe = torch.tensor(cwe_to_cwe)
        cwe_to_capec = torch.tensor(cwe_to_capec)
        cve_to_cwe = torch.tensor(cve_to_cwe)
        
        self.data["capec", "to", "capec"].edge_index = torch.stack([capec_to_capec[:, 0], capec_to_capec[:, 2]])
        self.data["capec", "to", "capec"].edge_label = capec_to_capec[:, 1]

        self.data["cwe", "to", "cwe"].edge_index = torch.stack([cwe_to_cwe[:, 0], cwe_to_cwe[:, 2]])
        self.data["cwe", "to", "cwe"].edge_label = cwe_to_cwe[:, 1]

        self.data["cwe", "to", "capec"].edge_index = torch.stack([cwe_to_capec[:, 0], cwe_to_capec[:, 2]])
        self.data["cwe", "to", "capec"].edge_label = cwe_to_capec[:, 1]

        self.data["cve", "to", "cwe"].edge_index = torch.stack([cve_to_cwe[:, 0], cve_to_cwe[:, 2]])
        self.data["cve", "to", "cwe"].edge_label = cve_to_cwe[:, 1]
        
    def split_data(self):
        transform = T.RandomLinkSplit(
            num_val=0.05,
            num_test=0.1,
            neg_sampling_ratio=1.0,
            is_undirected=False,
            add_negative_train_samples=False,
            edge_types=[
                ("capec", 'to', "capec"),
                ("cwe", 'to', "cwe"),
                ("cwe", 'to', "capec"),
                ("cve", 'to', "cwe")
            ],
        )
        
        train_data, valid_data, test_data = transform(self.data)
        
        return train_data, valid_data, test_data
        
            
    
        

if __name__=="__main__":
    create_data = CreateData()
    
    train_data, valid_data, test_data = create_data.split_data()
    
    # print(train_data["cwe", "to", "capec"].edge_label_index)
    
    # print(valid_data["cwe", "to", "capec"].edge_label_index)
    
    print(train_data)
    
    print(valid_data)
    
    print(test_data)
    
    test_edge_index = test_data["cwe", "to", "capec"].edge_label_index
    train_edge_index = train_data["cwe", "to", "capec"].edge_label_index
    
    train_edge_index = train_edge_index.to('cpu').detach().numpy().copy().T.tolist()
    test_edge_index = test_edge_index.to('cpu').detach().numpy().copy().T.tolist()
    
    print(len(train_edge_index))
    
    print(len(test_edge_index))
    
    count = 0
    for edge in test_edge_index:
        if edge in train_edge_index:
            count += 1
    
    print(count)
            