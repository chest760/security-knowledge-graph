import torch
from .base import TextEmbedding
from transformers import AutoTokenizer, AutoModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SentenceTransformer(TextEmbedding):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        
    def text_to_ids(
        self, 
        text_list: list[str]
    ):
        ids = []
        masks = []
        for text in text_list:
            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=250,
                padding='max_length',
                truncation=True
            )
            ids.append(inputs['input_ids'])
            masks.append(inputs['attention_mask'])

        return {"id": torch.LongTensor(ids), "mask": torch.LongTensor(masks)}
        
    def forward(
        self,
        text_list: list[str]
    ):  
        result = self.text_to_ids(text_list)
        id = result["id"].to(device)
        mask = result["mask"].to(device)
        output = self.model(id, mask, output_hidden_states=True)
        return output.pooler_output
    
        