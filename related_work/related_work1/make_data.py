import torch
import pandas as pd


dataset = pd.read_csv("./data/data.csv")
pair = pd.read_csv("./data/pair.csv")

mapped_id = pd.DataFrame(
    {
        "ID": dataset["ID"],
        "Description": dataset["Description"],
        "mapped_id": range(len(dataset))
    }
)

mapped_relation = pd.DataFrame(
    {
        "Relation": pair["Relation"].unique().tolist(),
        "mapped_id": range(len(pair["Relation"].unique().tolist()))
    }
)

print(mapped_relation)

def create_tuple():
    tuples = []
    for tuple in pair.to_dict(orient="records"):
        try:
            id1 = tuple["ID1"]
            id2 = tuple["ID2"]
            relation = tuple["Relation"]
            id1: int = mapped_id[mapped_id["ID"] == id1]["mapped_id"].item()
            id2: int = mapped_id[mapped_id["ID"] == id2]["mapped_id"].item()
            relation: int = mapped_relation[mapped_relation["Relation"] == relation]["mapped_id"].item()

            tuples.append([id1, relation, id2])
        except: 
            print(id1)
            print(id2)
    
    return torch.tensor(tuples)

if __name__ == "__main__":
    tuples = create_tuple()
    
    
    print(tuples)
    print(len(tuples))
        
        