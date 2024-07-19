import torch
import pandas as pd
from models.text_embedding.sentence_transformer import SentenceTransformer
from models.graph_embedding.transe import transE
from torch_geometric.data import HeteroData

class CreateHeteroData:
    def __init__(
        self,
        triplet: pd.DataFrame,
        dataset: pd.DataFrame,
        text_embedding_model: SentenceTransformer,
        graph_embedding: transE
    ):
        self.hetero_data = HeteroData()
        self.text_embedding_model = text_embedding_model
        self.graph_embedding = graph_embedding
        self.dataset = dataset
        self.triplet = triplet
        
    
    @torch.nograd()
    def _get_text_embedding(self):
        embs = []
        for series in self.dataset.to_dict(orient="records"):
            name = series["Name"] if isinstance(series["Name"], str) else ""
            description = series["Description"] if isinstance(series["Description"], str) else ""
            text = name + description
            emb = self.text_embedding_model.forward(
                text_list=[text]
            )
            embs.append(emb)
        
        return torch.cat(embs, dim=1)
            
    
    @torch.nograd()
    def _get_graph_embedding(self):
        self.graph_embedding.forward(
            
        )
    
    def init_graph(self):
        text_embedding = self._get_graph_embedding()
        graph_embedding = self._get_graph_embedding()
        
        
        
        