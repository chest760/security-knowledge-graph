import sys # noqa
sys.path.append("../") # noqa
import os
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from embedding import VoyageAI
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

root_path = os.path.join(os.path.dirname(__file__), "../../")


class Clustering:
    def __init__(
        self,
        data: pd.DataFrame
    ) -> None:
        self.data = data
        embs = self.get_embedding()
        x = embs.detach().numpy().copy() # (3529, 1024)
        self.pca = PCA(n_components=2, random_state=42)
        x = self.pca.fit_transform(x)
        
        self.x = x
        
        self.hierarchy()
    
    def get_embedding(self) -> torch.tensor:
        client = VoyageAI()
        embs = []
        for series in self.data.to_dict(orient="records"):
            id = series["ID"]
            emb = client.read_embedding(id=id)
            embs.append(emb)
        
        return torch.stack(embs)
    
    
    def kmeans(self, n_cluster: int):
        kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        kmeans.fit(self.x)
        # クラスタ中心
        print("Cluster centers:", kmeans.cluster_centers_)

        # 各サンプルのクラスタラベル
        print("Labels:", kmeans.labels_)

        # イナーシャ（クラスタ内のコヒーレンス）
        print("Inertia:", kmeans.inertia_)

        # 反復回数
        print("Number of iterations:", kmeans.n_iter_)
    
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
    
        # 3Dプロットの作成
        fig = plt.figure()
        fig = plt.figure()
        ax = Axes3D(fig, elev=10, azim=270)
        fig.add_axes(ax)

        # 各クラスタのデータポイントをプロット
        scatter = ax.scatter(self.x[:, 0], self.x[:, 1], self.x[:, 2], c=labels, cmap='viridis', marker='o', alpha=0.5)
        # クラスタの中心をプロット
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=300, c='red', marker='x')

        # タイトルとラベル
        ax.set_title('3D Cluster Visualization')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')

        # 凡例の追加
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)

        # 表示
        plt.savefig("./b.png")
    
    def hierarchy(self):
        Z = linkage(self.x, 'ward')
        
        # デンドログラムの表示
        plt.figure(figsize=(50, 50))
        dendrogram(Z)
        plt.title('Dendrogram')
        plt.ylabel('Euclidean distance')
        plt.savefig("./c.png")

        num_clusters = 10
        clusters = fcluster(Z, num_clusters, criterion='maxclust')

        # クラスタリング結果のプロット
        plt.figure(figsize=(10, 8))
        colors = ['red', 'green', 'blue', 'yellow', 'pink', 'gold', 'magenta', 'orange', 'aqua', 'violet']
        for i in range(1, num_clusters+1):
            plt.scatter(self.x[clusters == i, 0], self.x[clusters == i, 1], color=colors[i-1], label=f'Cluster {i}')

        plt.title('Hierarchical Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.savefig("./d.png")
        
        
        
    
if __name__ == "__main__":
    data = pd.read_csv(f"{root_path}/data/processed/dataset_voyage.csv")[:559]
    Clustering(data)