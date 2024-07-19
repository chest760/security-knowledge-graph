import os
import torch
import pandas as pd

root_path = os.path.join(os.path.dirname(__file__), "../")

class Execute:
    def __init__(self):
        triplet = pd.read_csv(f"{root_path}/data/processed/triplet.csv")[:722]
        dataset = pd.read_csv(f"{root_path}/data/processed/dataset.csv")[:559]
    
    def init_data(self):
        
    
    def train():
        pass
    
    @torch.no_grad()
    def valid():
        pass

    @torch.no_grad()
    def test():
        pass



    
    

if __name__ == "__main__":
    main()