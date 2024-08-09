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
from sklearn.cluster import HDBSCAN
import hdbscan

root_path = os.path.join(os.path.dirname(__file__), "../../")
colors = [
    '#FF0000',  # Red
    '#00FF00',  # Lime
    '#0000FF',  # Blue
    '#FFFF00',  # Yellow
    '#FF00FF',  # Magenta
    '#00FFFF',  # Cyan
    '#800000',  # Maroon
    '#008000',  # Green
    '#000080',  # Navy
    '#808000',  # Olive
    '#800080',  # Purple
    '#008080',  # Teal
    '#FFA500',  # Orange
    '#A52A2A',  # Brown
    '#FFC0CB',  # Pink
    '#E6E6FA',  # Lavender
    '#FFD700',  # Gold
    '#C0C0C0',  # Silver
    '#228B22',  # Forest Green
    '#4B0082',  # Indigo
    '#FF4500',  # Orange Red
    '#2E8B57',  # Sea Green
    '#8A2BE2',  # Blue Violet
    '#20B2AA',  # Light Sea Green
    '#FF69B4',  # Hot Pink
    '#00CED1',  # Dark Turquoise
    '#FF1493',  # Deep Pink
    '#00FA9A',  # Medium Spring Green
    '#1E90FF',  # Dodger Blue
    '#B22222',  # Fire Brick
]

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
        
        # embs_g = self.get_graph_embedding()
        
        embs = self.get_embedding()
        
        # embs = torch.cat([embs, embs_g], dim=1)
        
        x = embs.detach().numpy().copy() # (3529, 1024)
        # self.pca = PCA(n_components=3, random_state=42, tol=0.1)
        # x = self.pca.fit_transform(x)

        mapper = umap.UMAP(random_state=42,  metric="cosine",
                           n_neighbors=9, n_components=2, min_dist=0.4)
        x = mapper.fit_transform(x)
        
        self.x = x
    
    def get_embedding(self) -> torch.tensor:
        client = VoyageAI()
        embs = []
        for series in self.data.to_dict(orient="records"):
            id = series["ID"]
            emb = client.read_embedding(id=id)
            
            id = int(id[5:])
            names = []
            # names = self.domain_mechanism[self.domain_mechanism["ID"] == id]["Mechanism"].item().split(",")
            names.extend(self.domain_mechanism[self.domain_mechanism["ID"] == id]["Mechanism"].item().split(","))
            
            category_emb = 0
            for name in names:
                category=name.strip()
                category_emb += client.read_category_embedding(category)
            
            emb = torch.cat([emb, category_emb])
            
            embs.append(emb)
        
        return torch.stack(embs)

    def get_graph_embedding(self):
        model = RotatE(node_num=559, relation_num=5,hidden_channels=256, margin=2, p_norm=2)
        model.load_state_dict(torch.load("../models/graph_embedding/kge.pth"))
        
        re_emb = model.node_emb.weight
        im_emb = model.node_emb_im.weight
        
        return re_emb + im_emb
        
    
    
    def kmeans(self, n_cluster: int):
        kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        kmeans.fit(self.x)
    
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        print(labels)
    
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
        dendrogram(Z, labels=self.data["ID"].tolist())
        plt.title('Dendrogram')
        plt.ylabel('Euclidean distance')
        plt.savefig("./a.png")

        num_clusters = 7
        clusters = fcluster(Z, num_clusters, criterion='maxclust')

        # クラスタリング結果のプロット
        plt.figure(figsize=(10, 8))
        # colors = ['red', 'green', 'blue', 'yellow', 'pink', 'gold', 'magenta', 'orange', 'aqua', 'violet', 'lavender', 'tan', 'peru', 'palegreen', 'deepskyblue']
        dic = {
            i + 1: [] for i in range(num_clusters)
        }
        for i in range(1, num_clusters+1):
            l = [i for i, x in enumerate(clusters == i) if x]
            dic[i] = l
            plt.scatter(self.x[clusters == i, 0], self.x[clusters == i, 1], color=colors[i-1], label=f'Cluster {i}')

        plt.title('Hierarchical Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.savefig("./b.png")
        
        print(dic)
        
        return self.x, dic
    
    def hdbscan(self):
        # HDBSCANモデルを作成し、フィッティング
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1)
        cluster_labels = clusterer.fit_predict(self.x)
        
        # clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        # plt.savefig("./d.png")
        
        label_num = len(set(cluster_labels))

        # 結果をプロット
        plt.figure(figsize=(10, 7))
        for i in range(label_num):
            plt.scatter(self.x[cluster_labels==i-1, 0], self.x[cluster_labels==i-1, 1], c=colors[i], label=f'Cluster {i-1}')
        plt.title('HDBSCAN Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.savefig("./c.png")

        # クラスタリング結果を表示
        l = []
        dic = {
            i: [] for i in range(label_num)
        }
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        for index, i in enumerate(list(cluster_labels)):
            if i != -1:
                dic[i].append(index)
        n_noise = list(cluster_labels).count(-1)
        print(f'推定されたクラスタ数: {n_clusters}')
        print(f'ノイズとして分類されたポイント数: {n_noise}')
        
        return self.x, dic
    
    
    def run(self):
        # self.kmeans(8)
        
        x, dic = self.hierarchy()
        
        # x, dic = self.hdbscan()
        
        return x, dic
        
        
    
if __name__ == "__main__":
    data = pd.read_csv(f"{root_path}/data/processed/dataset_voyage.csv")[:559]
    capec_category = pd.read_csv(f"{root_path}/data/processed/capec_category_voyage.csv")
    capec_domain_mechanism = pd.read_csv(f"{root_path}/data/raw/capec_domain_mechanism.csv")
    Clustering(
        data=data,
        categoty=capec_category,
        domain_mechanism=capec_domain_mechanism
    ).run()