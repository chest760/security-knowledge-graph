import sys # noqa
sys.path.append("../../") # noqa
import os
import torch
import pandas as pd
from typing import Union
from data_process.create_positive_triples import (
    create_capec_triples,
    create_capec_cwe_triples,
    create_cwe_cve_triples,
    create_cwe_triples,
    Capec,
    Cwe,
    Cve,
    rel_type
)


root = os.path.join(os.path.dirname(__file__), "../../")

# データの読み込み
capec = pd.read_csv(f"{root}/data/raw/capec.csv")
cwe = pd.read_csv(f"{root}/data/raw/cwe.csv")
cve = pd.read_csv(f"{root}/data/raw/cve.csv")

# IDを一意に設定
capec["ID"] = ["CAPEC-" + str(id) for id in capec["ID"]]
cwe["ID"] = ["CWE-" + str(id) for id in cwe["CWE-ID"]]
cve["ID"] = [str(id) for id in cve["CVE-ID"]]

# IDを数字にマッピング
dataset = pd.concat([capec["ID"], cwe["ID"], cve["ID"]]).reset_index(drop=True)
mapped_id = pd.DataFrame(
    data={
        "ID": dataset,
        "mappedID": range(len(dataset))
    }
)


mapped_relation = {
    "ParentOf": 0,
    "ChildOf": 1,
    "CanPrecede": 2,
    "CanFollow": 3,
    "PeerOf": 4,
    "TargetOf": 5,
    "AttackOf": 6,
    "ExampleOf": 7
}


def get_mapped_id(
    id: Union[Capec, Cwe, Cve]
) -> int:
    if isinstance(id, Capec):
        id = "CAPEC-"+str(id.value)
    elif isinstance(id, Cwe):
        id = "CWE-"+str(id.value)
    else:
        id: str = str(id.value)
        
    
    return mapped_id[mapped_id["ID"] == id]["mappedID"].item()


def create_postive_triples() -> torch.tensor:
    triples = []
    capec_triples = create_capec_triples()
    cwe_triples = create_cwe_triples()
    cwe_capec_triples = create_capec_cwe_triples()
    cve_cwe_triples = create_cwe_cve_triples()
    
    
    for id1, rel, id2 in capec_triples:
        try:
            id1 = get_mapped_id(id1)
            id2 = get_mapped_id(id2)
            rel = mapped_relation[rel]
        except:
            continue
        triples.append([id1, rel, id2])
    
    capec_rel_num = len(triples)

    for id1, rel, id2 in cwe_triples:
        try:
            id1 = get_mapped_id(id1)
            id2 = get_mapped_id(id2)
            rel = mapped_relation[rel]
        except:
            continue
        triples.append([id1, rel, id2])
        
    cwe_rel_num = len(triples) - capec_rel_num

    for id1, rel, id2 in cwe_capec_triples:
        id1 = get_mapped_id(id1)
        id2 = get_mapped_id(id2)
        rel = mapped_relation[rel]
        triples.append([id1, rel, id2])
    
    cwe_capec_rel_num = len(triples) - capec_rel_num - cwe_rel_num

    for id1, rel, id2 in cve_cwe_triples:
        id1 = get_mapped_id(id1)
        id2 = get_mapped_id(id2)
        rel = mapped_relation[rel]
        triples.append([id1, rel, id2])
    
    cve_cwe_rel_num = len(triples) - capec_rel_num - cwe_rel_num - cwe_capec_rel_num
    
    return torch.tensor(triples)
    

def add_reverse_edge(triples: torch.Tensor) -> torch.Tensor:
    new_triples = []
    for triple in triples:
        id1 = triple[0].item()
        rel = triple[1].item()
        id2 = triple[2].item()
        if rel == 0:
            new_triples.append([id1, 0, id2])
            new_triples.append([id2, 1, id1])
        elif rel == 1:
            new_triples.append([id1, 1, id2])
            new_triples.append([id2, 0, id1])
        elif rel == 2:
            new_triples.append([id2, 2, id1])
            new_triples.append([id1, 3, id2])
        elif rel == 3:
            new_triples.append((id1, 3, id2))
            new_triples.append((id2, 2 ,id1))
        elif rel == 4:
            new_triples.append([id1, 4, id2])
            new_triples.append([id2, 4, id1])
        elif rel == 5:
            new_triples.append([id1, 5, id2])
            new_triples.append([id2, 6, id1])
        elif rel == 6:
            new_triples.append([id1, 6, id2])
            new_triples.append([id2, 5, id1])
    
    return torch.tensor(new_triples)



       
    
    
    
    
    
    