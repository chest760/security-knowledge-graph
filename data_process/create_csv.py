import re
import pandas as pd
from typing import Union

capec = pd.read_csv("../data/raw/capec.csv")
cwe = pd.read_csv("../data/raw/cwe.csv")
cve = pd.read_csv("../data/raw/cve.csv")

cve["Name"] = ""


def create_triples():
    triples = []
    check_dup = []
    for series in capec.to_dict(orient="records"):
        relations: Union[str, float] = series["Related Attack Patterns"]
        if isinstance(relations, float):
            continue
        relations = re.findall(r"::NATURE:(\w+):CAPEC ID:(\d+)", relations)
        for relation, id  in relations:
            if (str(id), str(series["ID"])) in check_dup or (str(series["ID"]), str(id)) in check_dup:
                continue
            if relation == "ChildOf":
                triples.append(["capec"+str(id),"capec"+str(series["ID"]), "ParentOf"])
            elif relation == "CanFollow":
                triples.append(["capec"+str(id),"capec"+str(series["ID"]), "CanPrecede"])
            elif relation == "ParentOf" or relation == "CanPrecede" or relation == "PeerOf":
                triples.append(["capec"+str(series["ID"]),"capec"+str(id), relation])

            check_dup.append((str(id), str(series["ID"])))
            check_dup.append((str(series["ID"]), str(id)))


    check_dup = []
    for series in cwe.to_dict(orient="records"):
        relations: Union[str, float] = series["Related Weaknesses"]
        if isinstance(relations, float):
            continue
        relations = re.findall(r"::NATURE:(\w+):CWE ID:(\d+):", relations)
        for relation, id  in relations:
            if (str(id), str(series["CWE-ID"])) in check_dup or (str(series["CWE-ID"]), str(id)) in check_dup:
                continue
            if relation == "ChildOf":
                triples.append(["cwe"+str(id),"cwe"+str(series["CWE-ID"]), "ParentOf"])
            elif relation == "CanFollow":
                triples.append(["cwe"+str(id),"cwe"+str(series["CWE-ID"]), "CanPrecede"])
            elif relation == "ParentOf" or relation == "CanPrecede" or relation == "PeerOf":
                triples.append(["cwe" + str(series["CWE-ID"]),"cwe"+str(id), relation])

            check_dup.append((str(id), str(series["CWE-ID"])))
            check_dup.append((str(series["CWE-ID"]), str(id)))

    check_dup = []
    for series in capec.to_dict(orient="records"):
        capec_id: int = series["ID"]
        weaknesses: Union[str, float] = series["Related Weaknesses"]
        if isinstance(weaknesses, float):
            continue
        weaknesses: list[str] = re.findall(r"::(\d+)", weaknesses)
        for cwe_id in weaknesses:
            if ("cwe"+str(cwe_id), "capec"+str(capec_id)) in check_dup:
                continue
            triples.append(["cwe"+str(cwe_id), "capec"+str(capec_id), "TargetOf"])   
            check_dup.append(("cwe"+str(cwe_id), "capec"+str(capec_id)))

    check_dup = []
    for series in cwe.to_dict(orient="records"):
        cwe_id: int = series["CWE-ID"]
        examples: Union[str, float] = series["Observed Examples"]
        if isinstance(examples, float):
            continue
        examples: list[str] = re.findall(r"::REFERENCE:(CVE-\d+-\d+):DESCRIPTION:([^:]+(?::(?!:)[^:]+)*)", examples)
        for cve_id, description in examples:
            if (str(cve_id),"cwe"+str(cwe_id)) in check_dup:
                continue
            triples.append([str(cve_id),"cwe"+str(cwe_id), "InstanceOf"])
            check_dup.append((str(cve_id),"cwe"+str(cwe_id)))

    df = pd.DataFrame(data=triples, columns=["ID1", "ID2", "Relation"])
    df.to_csv("../data/processed/triplet.csv")

def create_dataset():
    cwe["ID"] = ["cwe"+str(id) for id in cwe["CWE-ID"].unique().tolist()]
    capec["ID"] = ["capec"+str(id) for id in capec["ID"].unique().tolist()]
    dataset = pd.concat([capec[["ID", "Name", "Description"]], cwe[["ID", "Name", "Description"]], cve[["ID", "Name", "Description"]]]).reset_index(drop=True)
    dataset.to_csv("../data/processed/dataset.csv")

create_triples()