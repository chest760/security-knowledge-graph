import os
import torch
from typing import Union
import voyageai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

root_path = os.path.join(os.path.dirname(__file__), "../")
dataset = pd.read_csv(f"{root_path}/data/processed/dataset.csv")

class Base:
    data = pd.read_csv(f"{root_path}/data/processed/dataset_voyage.csv")
    
    def read_embedding(self, id: Union[int, str]):
        selected_row = self.data[self.data["ID"]==id]
        emb = selected_row["Embedding"].apply(eval).apply(torch.tensor).item()
        return emb

class VoyageAI(Base):
    def __init__(self):
        self.client = voyageai.Client(api_key=os.getenv("VOYAGE_AI_API_KEY"))
    
    def get(self, text: list[str]):
        result = self.client.embed(
            text, 
            model="voyage-large-2-instruct",
            input_type="document"
        )
        
        return result.embeddings[0]
    
    def __call__(self):
        l = []
        for series in dataset.to_dict(orient="records"):
            id = series["ID"]
            name = series["Name"] if isinstance(series["Name"], str) else ""
            description = series["Description"] if isinstance(series["Description"], str) else ""
            text = name + description
            emb = self.get([text])
            l.append([id, name, description, emb])
            print(id)
        df = pd.DataFrame(l, columns=['ID','Name', 'Description', 'Embedding'])
        df.to_csv(f"{root_path}/data/processed/dataset_voyage.csv", index=False)
            
            
class OpenAI:
    pass


if __name__ == "__main__":
    voyage = VoyageAI()
    voyage()