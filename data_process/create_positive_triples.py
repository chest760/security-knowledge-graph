import os
import re
import math
import numpy as np
import pandas as pd
from typing import Union, Literal, NewType

root = os.path.join(os.path.dirname(__file__), "../")

capec = pd.read_csv(f"{root}/data/raw/capec.csv")
cwe = pd.read_csv(f"{root}/data/raw/cwe.csv")

class Cwe:
    def __init__(self, value: Union[str, int]):
        self.value = value

class Capec:
    def __init__(self, value: Union[str, int]):
        self.value = value

class Cve:
    def __init__(self, value: Union[str, int]):
        self.value = value
    


rel_type = Literal[
    "ParentOf",
    "ChildOf",
    "CanPrecede",
    "CanFollow",
    "PeerOf",
    "TragetOf",
    "AttackOf",
    "InstanceOf",
    "AbstractionOf"
]

def isnan(input: any) -> bool:
    if isinstance(input, (int, float)):
        return math.isnan(input)
    else:
        return isinstance(input, float)


def delete_duplicate(triples: list[tuple[Union[Capec, Cve], rel_type, Union[Capec,Cve]]]):
    new_triples = []
    tmp_pair = []
    for triple in triples:
        id1 = triple[0]
        id2 = triple[2]
        if (id1.value, id2.value) not in tmp_pair:
            tmp_pair.append((id1.value, id2.value))
            tmp_pair.append((id2.value, id1.value))
            new_triples.append(triple)
    
    return new_triples


def create_capec_triples() -> list[tuple[Capec, rel_type, Capec]]:
    triples = []
    for data in capec.to_dict(orient="records"):
        data_id: int = data["ID"]
        relations: Union[str, float] = data["Related Attack Patterns"]
        if isnan(relations):
            continue
        relations: list[str] = re.findall(r"::NATURE:(\w+):CAPEC ID:(\d+)", relations)
        for rel_type, id in relations:
            triples.append((Capec(data_id), rel_type, Capec(int(id))))
    triples = delete_duplicate(triples)
    return triples


def create_cwe_triples() -> list[tuple[Cwe, rel_type, Cwe]]:
    triples = []
    for data in cwe.to_dict(orient="records"):
        data_id: int = data["CWE-ID"]
        relations: Union[str, float] = data["Related Weaknesses"]
        if isnan(relations):
            continue
        relations: list[str] = re.findall(r"::NATURE:(\w+):CWE ID:(\d+):", relations)
        for rel_type, id in relations:
            triples.append((Cwe(data_id), rel_type, Cwe(int(id))))
    delete_duplicate(triples)
    return triples


def create_capec_cwe_triples() -> list[tuple[Cwe, rel_type, Capec]]:
    triples = []
    for data in capec.to_dict(orient="records"):
        capec_id: int = data["ID"]
        weaknesses: Union[str, float] = data["Related Weaknesses"]
        if isnan(weaknesses):
            continue
        weaknesses: list[str] = re.findall(r"::(\d+)", weaknesses)
        for cwe_id in weaknesses:
            triples.append((Cwe(int(cwe_id)), "TargetOf", Capec(capec_id)))
    return triples


def create_cwe_cve_triples() -> list[tuple[Cve, rel_type, Cwe]]:
    triples = []
    cves = []
    cve_ids = []
    for data in cwe.to_dict(orient="records"):
        cwe_id: int = data["CWE-ID"]
        examples: Union[str, float] = data["Observed Examples"]
        if isnan(examples):
            continue
        examples: list[str] = re.findall(r"::REFERENCE:(CVE-\d+-\d+):DESCRIPTION:([^:]+(?::(?!:)[^:]+)*)", examples)
        for cve_id, description in examples:
            triples.append((Cve(cve_id), "InstanceOf", Cwe(str(cwe_id))))
            if cve_id not in cve_ids:  
               cves.append([cve_id, description])
               cve_ids.append(cve_id)
    cve_df = pd.DataFrame(data=cves, columns=["CVE-ID", "Description"])
    cve_df.to_csv(f"{root}/data/raw/cve.csv")
    return triples

