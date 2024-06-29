import torch
from .base import TextEmbedding
from transformers import AutoTokenizer, AutoModel

class SentenceTransformer(TextEmbedding):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        
    def forward(
        self,
        id:torch.LongTensor,
        mask:torch.LongTensor
    ):
        output = self.model(id, mask, output_hidden_states=True)
        return output.pooler_output
    
        