import sys # noqa
sys.path.append("../") # noqa
import os
import umap
import torch
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from embedding import VoyageAI
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from models.graph_embedding.rotate import RotatE

root_path = os.path.join(os.path.dirname(__file__), "../../")


class Clustering:
    def __init__(
        self,
        data: pd.DataFrame,
        categoty: pd.DataFrame,
        domain_mechanism: Optional[pd.DataFrame]
    ) -> None:
        self.data = data
        self.category = categoty
        self.domain_mechanism = domain_mechanism
        
        embs = self.get_graph_embedding()
        
        # embs = self.get_embedding()
        
        # embs = torch.cat([embs, embs_g], dim=1)
        
        x = embs.detach().numpy().copy() # (3529, 1024)
        # self.pca = PCA(n_components=3, random_state=42, tol=0.1)
        # x = self.pca.fit_transform(x)

        mapper = umap.UMAP(random_state=42,  metric="cosine",
                           n_neighbors=10, n_components=2, min_dist=0.1)
        x = mapper.fit_transform(x)
        
        self.x = x
        
        self.kmeans(9)
    
    def get_embedding(self) -> torch.tensor:
        client = VoyageAI()
        embs = []
        for series in self.data.to_dict(orient="records"):
            id = series["ID"]
            emb = client.read_embedding(id=id)
            
            # id = int(id[5:])
            # names = self.domain_mechanism[self.domain_mechanism["ID"] == id]["Mechanism"].item().split(",")
            # # names.extend(self.domain_mechanism[self.domain_mechanism["ID"] == id]["Mechanism"].item().split(","))
            
            # category_emb = 0
            # for name in names:
            #     category_emb += client.read_category_embedding(category=name.strip())
            
            # emb = torch.cat([emb, 0.5*category_emb/len(names)])
            
            embs.append(emb)
        
        return torch.stack(embs)

    def get_graph_embedding(self):
        model = RotatE(node_num=559, relation_num=7,hidden_channels=256, margin=2, p_norm=2)
        model.load_state_dict(torch.load("../models/graph_embedding/kge.pth"))
        
        re_emb = model.node_emb.weight
        im_emb = model.node_emb_im.weight
        
        return (re_emb * im_emb) / 2
        
    
    
    def kmeans(self, n_cluster: int):
        kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        kmeans.fit(self.x)
    
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
    
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # 各クラスタのデータポイントをプロット
        scatter = ax.scatter(self.x[:, 0], self.x[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
        # クラスタの中心をプロット
        ax.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='x')

        # タイトルとラベル
        ax.set_title('Cluster Visualization')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')


        # 凡例の追加
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)

        # 表示
        plt.savefig("./kmeans1.png")
    
    def hierarchy(self):
        Z = linkage(self.x, 'ward')
        
        # デンドログラムの表示
        plt.figure(figsize=(50, 50))
        dendrogram(Z)
        plt.title('Dendrogram')
        plt.ylabel('Euclidean distance')
        plt.savefig("./a.png")

        num_clusters = 9
        clusters = fcluster(Z, num_clusters, criterion='maxclust')

        # クラスタリング結果のプロット
        plt.figure(figsize=(10, 8))
        colors = ['red', 'green', 'blue', 'yellow', 'pink', 'gold', 'magenta', 'orange', 'aqua', 'violet', 'lavender', 'tan', 'peru', 'palegreen', 'deepskyblue']
        for i in range(1, num_clusters+1):
            plt.scatter(self.x[clusters == i, 0], self.x[clusters == i, 1], color=colors[i-1], label=f'Cluster {i}')

        plt.title('Hierarchical Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.savefig("./b.png")
        
        
        
    
if __name__ == "__main__":
    data = pd.read_csv(f"{root_path}/data/processed/dataset_voyage.csv")[:559]
    capec_category = pd.read_csv(f"{root_path}/data/processed/capec_category_voyage.csv")
    capec_domain_mechanism = pd.read_csv(f"{root_path}/data/raw/capec_domain_mechanism.csv")
    Clustering(
        data=data,
        categoty=capec_category,
        domain_mechanism=capec_domain_mechanism
    )