import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt

triple = pd.read_csv("../../data/processed/triplet.csv")[:722]
dataset = pd.read_csv("../../data/processed/dataset.csv")[:559]

def change_index(data:pd.DataFrame, triplets_df: pd.DataFrame):
    mapped_id = pd.DataFrame(
        data={
            "ID": data["ID"],
            "Name": data["Name"],
            "Description": data["Description"],
            "mappedID": range(len(data))
        }
    )
    
    triplets = []
    for triplet in triplets_df.to_dict(orient="records"):
        id1 = mapped_id[mapped_id["ID"] == triplet["ID1"]]["mappedID"].item()
        id2 = mapped_id[mapped_id["ID"] == triplet["ID2"]]["mappedID"].item()
        
        triplets.append((id1, id2))
    return triplets


triplet = change_index(dataset, triple)
print(triplet)

G = nx.Graph()

G.add_edges_from(triplet)

print(G)

# Louvainアルゴリズムを使用してコミュニティを検出
partition = community_louvain.best_partition(G)

# 結果の分析
community_sizes = {}
for node, community_id in partition.items():
    if community_id not in community_sizes:
        community_sizes[community_id] = 0
    community_sizes[community_id] += 1

print(f"検出されたコミュニティ数: {len(community_sizes)}")
print(f"最大のコミュニティサイズ: {max(community_sizes.values())}")
print(f"最小のコミュニティサイズ: {min(community_sizes.values())}")