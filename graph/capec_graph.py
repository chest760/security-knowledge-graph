import os
import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

root_path = os.path.join(os.path.dirname(__file__), "../")

class Graph:
    def __init__(
        self,
        triplet: pd.DataFrame
    ) -> None:
        self.triplet = triplet
        self.rel_dict={
            "ParentOf": [],
            "CanPrecede": [],
            "PeerOf": []
        }
        
        self.rel_color_dict={
            "ParentOf": "red",
            "CanPrecede": "green",
            "PeerOf": "blue"
        }
        self.G = nx.DiGraph()
        
        self.init_graph()
        
    
    def init_graph(self):
        self.make_edge()

    def make_edge(self):
        edge_colors = []
        all_edges = []
        for series in triplet.to_dict(orient="records"):
            id1 = int(series["ID1"][5:])
            id2 = int(series["ID2"][5:])
            rel = series["Relation"]
            edge_colors.append(self.rel_color_dict[rel])
            all_edges.append((id1, id2))
        
        self.G.add_edges_from(all_edges)
    
        self.edges = all_edges
        self.edge_colors = edge_colors
        
    
    def draw(self, node_pos: dict[str, np.array]):
        plt.figure(figsize=(70, 70))
        nx.draw_networkx(self.G, pos=node_pos,edgelist=self.edges, edge_color=self.edge_colors)
        plt.savefig("./graph.png")

class ClusteringGraph(Graph):
    def __init__(self) -> None:
        super().__init__()


class Domain(Graph):
    def __init__(
        self,
        domain: pd.DataFrame,
        triplet: pd.DataFrame
        
    ) -> None:
        super().__init__(
            triplet=triplet
        )
        
        self.domain_type = [
            "Software",
            "Hardware",
            "Communications",
            "Supply Chain",
            "Social Engineering",
            "Physical Security"
        ]
        
        self.domain = domain
        self.domain_ids_dict:dict[str, list] = {}
        self.id_domain_dict = {
            id: [] for id in self.G.nodes
        }
        self.node_pos_dict = {
            id: np.zeros(2) for id in self.G.nodes
        }
        
        for domain_df in self.domain.to_dict(orient="records"):
            id = domain_df["ID"]
            domains:str = domain_df["Domain"]
            
            if domains not in self.domain_ids_dict:
                self.domain_ids_dict[domains] = []
            
            
            self.domain_ids_dict[domains].append(id)
            self.id_domain_dict[id] = list(map(lambda x: x.strip(), domains.split(",")))
            
        
        print(self.domain_ids_dict)
            
        self.cluster()
        self.draw(self.node_pos_dict)
        
    def cluster(self):
        domain_pos_dict = {}
        
        for index, domain in enumerate(self.domain_type):
            rad = math.pi * 2 / len(self.domain_type)
            x = math.cos(rad * index) * 15
            y = math.sin(rad * index) * 15
            domain_pos_dict[domain] = np.array([x, y])
                    
        
        for index, (key, ids) in enumerate(self.domain_ids_dict.items()):
            rad = math.pi * 2 / len(ids)
            
            center = np.zeros(2)
            for domain in key.split(","):
                domain = domain.strip()
                center += domain_pos_dict[domain]
            
            center = center / len(key.split(","))
            
            scale = len(ids) if len(ids) < 100 else len(ids) * 0.3
            
            for index, id in enumerate(ids):
                try:
                    self.node_pos_dict[id] = center
                    x = math.cos(rad * index) * 0.03 * scale
                    y = math.sin(rad * index) * 0.03 * scale
                    self.node_pos_dict[id] = np.array([x, y]) + center
                except KeyError:
                    self.node_pos_dict[id] = center
                    self.node_pos_dict[id] = np.array([x, y]) + center

                
        
        
        

class Mechanism(Graph):
    def __init__(
        self,
        mechanism: pd.DataFrame,
        triplet: pd.DataFrame
        
    ) -> None:
        super().__init__(
            triplet=triplet
        )
        
        
        self.mechanism_type = [
            "Engage in Deceptive Interactions",
            "Collect and Analyze Information",
            "Employ Probabilistic Techniques",
            "Manipulate System Resources",
            "Manipulate Timing and State",
            "Inject Unexpected Items",
            "Manipulate Data Structures",
            "Abuse Existing Functionality",
            "Subvert Access Control",
        ]

        self.mechanism = mechanism
        self.mechanism_ids_dict:dict[str, list] = {}
        self.id_mechanism_dict = {
            id: [] for id in self.G.nodes
        }
        self.node_pos_dict = {
            id: np.zeros(2) for id in self.G.nodes
        }
        
        for mechanism_df in self.mechanism.to_dict(orient="records"):
            id = mechanism_df["ID"]
            mechanisms: str = mechanism_df["Mechanism"]
            
            if mechanisms not in self.mechanism_ids_dict:
                self.mechanism_ids_dict[mechanisms] = []
            
            
            self.mechanism_ids_dict[mechanisms].append(id)
            self.id_mechanism_dict[id] = list(map(lambda x: x.strip(), mechanisms.split(",")))
            
        self.cluster()
        self.draw(self.node_pos_dict)
        

    def cluster(self):
        mechanism_pos_dict = {}
        
        for index, mechanism in enumerate(self.mechanism_type):
            rad = math.pi * 2 / len(self.mechanism_type)
            x = math.cos(rad * index) * 15
            y = math.sin(rad * index) * 15
            mechanism_pos_dict[mechanism] = np.array([x, y])
                    
        
        for index, (key, ids) in enumerate(self.mechanism_ids_dict.items()):
            rad = math.pi * 2 / len(ids)
            
            center = np.zeros(2)
            for mechanism in key.split(","):
                mechanism = mechanism.strip()
                center += mechanism_pos_dict[mechanism]
            
            center = center / len(key.split(","))
            
            scale = len(ids) if len(ids) < 100 else len(ids) * 0.9
            
            for index, id in enumerate(ids):
                try:
                    self.node_pos_dict[id] = center
                    x = math.cos(rad * index) * 0.03 * scale
                    y = math.sin(rad * index) * 0.03 * scale
                    self.node_pos_dict[id] = np.array([x, y]) + center
                except KeyError:
                    self.node_pos_dict[id] = center
                    self.node_pos_dict[id] = np.array([x, y]) + center
        


def main(
    category: pd.DataFrame,
    triplet: pd.DataFrame
):
    # domain_graph= Domain(
    #     domain=category[["ID", "Domain"]],
    #     triplet=triplet   
    # )
    
    domain_graph= Mechanism(
        mechanism=category[["ID", "Mechanism"]],
        triplet=triplet   
    )
    
    


if __name__ == "__main__":
    category = pd.read_csv(f"{root_path}/data/raw/capec_domain_mechanism.csv")
    triplet = pd.read_csv(f"{root_path}/data/processed/triplet.csv")[:722]
    
    main(
        category=category,
        triplet=triplet
    )