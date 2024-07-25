import os
import pandas as pd
import networkx as nx

root_path = os.path.join(os.path.dirname(__file__), "../")
triplet = pd.read_csv(f"{root_path}/data/processed/triplet.csv")
dataset = pd.read_csv(f"{root_path}/data/processed/dataset.csv")




def create_graph(
    dataset: pd.DataFrame,
    triplet: pd.DataFrame,
    ):
    ids = []
    for series in triplet.to_dict(orient="records"):
        id1 = series["ID1"]
        id2 = series["ID2"]
        if series["Relation"] != "PeerOf":
            ids.append((id1, id2))
            
        # ids.append((id1, id2))
    G = nx.DiGraph()
    G.add_edges_from(ids)
    return G


def sort(cycle_with_relation: list[tuple[str, str, str]]):
    def search_connect_tail_and_head(relation: list[str, str, str], new_relations: list, count: int):
        count += 1
        new_relations.append(relation)
        tail_id = relation[2]
        next_relation = list(filter(lambda x: x[0]==tail_id, cycle_with_relation))
        if len(next_relation) == 0:
            l = [x for x in cycle_with_relation if x not in new_relations]
            return None
        if count == len(cycle_with_relation):
            return None
        search_connect_tail_and_head(next_relation[0], new_relations, count)
        return new_relations
        
    for i, relation in enumerate(cycle_with_relation):
        new_relations = []
        relations = search_connect_tail_and_head(relation, new_relations, count=0)
        if relations:
            break
    
    return relations
    

def check(cycle: list[str]):
    relations = []
    for i in range(len(cycle) - 1):
        id1 = cycle[i]
        id2 = cycle[i+1]
        series = triplet[(triplet["ID1"] == id1) & (triplet["ID2"] == id2)]
        if len(series) > 0:
            relations.append([series["ID1"].item(), series["Relation"].item(), series["ID2"].item()])
        else:
            series = triplet[(triplet["ID1"] == id2) & (triplet["ID2"] == id1)]
            relations.append([series["ID1"].item(), series["Relation"].item(), series["ID2"].item()])
    
    id1 = cycle[0]
    id2 = cycle[-1]
    series = triplet[(triplet["ID1"] == id1) & (triplet["ID2"] == id2)]
    if len(series) > 0:
        relations.append([series["ID1"].item(), series["Relation"].item(), series["ID2"].item()])
    else:
        series = triplet[(triplet["ID1"] == id2) & (triplet["ID2"] == id1)]
        relations.append([series["ID1"].item(), series["Relation"].item(), series["ID2"].item()])
    
    connected_list = sort(cycle_with_relation=relations)
    
    if connected_list is None:
        return None
    
    
    head_id = connected_list[0][0]
    tail_id = connected_list[-1][2]
    target_relation = None
    
    for target in relations:
        
        if target[0] == head_id and target[2] == tail_id:
            target_relation = target[1]
            break
        # 閉路
        elif head_id == tail_id:
            print(connected_list)
            return "Cycled"
    
    r = []
    for i in range(len(connected_list)):
        r.append(connected_list[i][1])
    
    if target_relation is None:
        return None

    
    if "ParentOf" in r and "CanPrecede" not in r and "PeerOf" not in r:
        if target_relation != "ParentOf":
            print(relations)
            return "Attention!!"

    if "CanPrecede" in r and "ParentOf" not in r and "PeerOf" not in r:
        if target_relation != "CanPrecede":
            print(relations)
            return "Attention!!"

    if "CanPrecede" in r and "PeerOf" in r and "ParentOf" not in r:
        if target_relation == "ParentOf":
            print(relations)
            return "Attention!!"

    if "ParentOf" in r and "PeerOf" in r and "CanPrecede" not in r:
        if target_relation == "CanPrecede":
            print(relations)
            return "Attention!!"
        
    
    return None
        
def main(
    dataset:pd.DataFrame,
    triplet: pd.DataFrame,
    ) -> None:
    
    G = create_graph(
        dataset=dataset,
        triplet=triplet,
    )
    
    # 閉路を調べる
    cycles = list(nx.simple_cycles(G))
    for a in cycles:
        if len(a) < 12 and len(a) > 2:
            print(a)
    
    cycles = list(nx.cycle_basis(G.to_undirected()))
    for cycle in cycles:
        a = check(cycle)
        if a:
            print(a)
        

if __name__ == "__main__":
    capec_triplet = triplet[722:2087]
    capec_dataset = dataset[559:1497]
    capec_triplet = triplet[:722]
    capec_dataset = dataset[:559]
    main(
        dataset=capec_dataset,
        triplet=capec_triplet,
    )