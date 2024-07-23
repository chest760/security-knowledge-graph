import os
import torch
import pandas as pd
from typing import Tuple

root_path = os.path.join(os.path.dirname(__file__), "../")

@torch.no_grad()
def random_sample(
    head_index: torch.Tensor, 
    rel_type: torch.Tensor,
    tail_index: torch.Tensor,
    num_node: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_negatives = head_index.numel() // 2
    rnd_index = torch.randint(num_node, head_index.size(),device=head_index.device)
    
    head_index = head_index.clone()
    head_index[:num_negatives] = rnd_index[:num_negatives]
    tail_index = tail_index.clone()
    tail_index[num_negatives:] = rnd_index[num_negatives:]

    return head_index, rel_type, tail_index

class Execute(torch.nn.Module):
    def __init__(
        self,
    ):
        triplet = pd.read_csv(f"{root_path}/data/processed/triplet.csv")[:722]
        dataset = pd.read_csv(f"{root_path}/data/processed/dataset.csv")[:559]
    
    def init_data(self):
        pass
        
    
    def train():
        pass
    
    @torch.no_grad()
    def valid():
        pass

    @torch.no_grad()
    def test():
        pass


def main():
    pass

    
    

if __name__ == "__main__":
    main()