import pandas as pd

triplet = pd.read_csv("../data/processed/triplet.csv")

triplets = []
for series in triplet.to_dict(orient="records"):
    id1 = series["ID1"]
    id2 = series["ID2"]
    rel = series["Relation"]
    
    if rel == "ParentOf":
        triplets.append([id2, id1, "ChildOf"])
    elif rel == "CanPrecede":
        triplets.append([id2, id1, "CanFollow"])
    elif rel == "PeerOf":
        triplets.append([id2, id1, "PeerOf"])
    elif rel == "TargetOf":
        triplets.append([id2, id1, "AttackOf"])
    

df = pd.DataFrame(data=triplets, columns=["ID1", "ID2", "Relation"])

df.to_csv("../data/processed/reverse_triplet.csv")