import sys
sys.path.append("../../")
import os
import torch
import pandas as pd
from typing import Tuple
from transe import TransE
from transh import TransH
from rotate import RotatE
from typing import Literal
import torch.nn.functional as F
from utils.triplet_loader import TripletLoader
from utils.static_seed import static_seed

root_path = os.path.join(os.path.dirname(__file__), "../../../")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

relations = ["ParentOf", "ChildOf", "CanPrecede", "CanFollow", "PeerOf"]
 
def change_index(data:pd.DataFrame, triplets_df: pd.DataFrame):
    mapped_id = pd.DataFrame(
        data={
            "ID": data["ID"],
            "Name": data["Name"],
            "Description": data["Description"],
            "mappedID": range(len(data))
        }
    )
    
    mapped_relation = pd.DataFrame(
        data={
            "Relation": relations,
            "relationID": range(len(relations))
        }
    )
    
    triplets = []
    for triplet in triplets_df.to_dict(orient="records"):
        id1 = mapped_id[mapped_id["ID"] == triplet["ID1"]]["mappedID"].item()
        id2 = mapped_id[mapped_id["ID"] == triplet["ID2"]]["mappedID"].item()
        relation = mapped_relation[mapped_relation["Relation"] == triplet["Relation"]]["relationID"].item()
        
        triplets.append([id1, relation, id2])
    return torch.tensor(triplets)

def reverse_triplet(triplets: torch.tensor):
    new_triplets = []
    for triplet in triplets:
        id1 = triplet[0].item()
        relation = triplet[1].item()
        id2 = triplet[2].item()
        new_triplets.append([id1, relation, id2])
        if relation == 0 or relation == 2 or relation == 5 or relation == 7:
            new_triplets.append([id2, relation+1, id1])
        elif relation == 4:
            new_triplets.append([id2, relation, id1])
            
    return torch.tensor(new_triplets)

def split_data(
    triplets: torch.Tensor,
    train: int=0.85,
    valid: int=0.05,
    test: int=0.1
) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    
    triplets = reverse_triplet(triplets=triplets)
    triple_num = len(triplets)
    static_seed(42)
    rnd_index = torch.randperm(triple_num)
    triplets = triplets[rnd_index]
    
    train_index = int(triple_num * train)
    valid_index = int(triple_num * valid)
    train_triples = triplets[:train_index]
    valid_triples = triplets[train_index:train_index+valid_index]
    test_triples = triplets[train_index+valid_index:]
    
    # train_triples = reverse_triplet(triplets=train_triples)
    # valid_triples = reverse_triplet(triplets=valid_triples)
    # test_triples = reverse_triplet(triplets=test_triples)
    
    return train_triples, valid_triples, test_triples

def data_loader(
    triplet: torch.tensor
):
    loader = TripletLoader(
        head_index=triplet[:, 0],
        rel_type=triplet[: ,1],
        tail_index=triplet[: ,2],
        batch_size=32
    )
    
    return loader

@torch.no_grad()
def random_sample(
    head_index: torch.Tensor, 
    rel_type: torch.Tensor,
    tail_index: torch.Tensor,
    num_node: int,
    seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_negatives = head_index.numel() // 2
    static_seed(seed)
    rnd_index = torch.randint(num_node, head_index.size(),device=head_index.device)
    
    head_index = head_index.clone()
    head_index[:num_negatives] = rnd_index[:num_negatives]
    tail_index = tail_index.clone()
    tail_index[num_negatives:] = rnd_index[num_negatives:]

    return head_index, rel_type, tail_index

class Exec:
    def __init__(
        self,
        model_type: Literal["transh", "transe", "rotate"],
        node_num: int,
        relaion_num: int,
        hidden_channels: int,
        p_norm: float = 2.0,
        margin: float = 2.0
    ) -> None:
        if model_type == "transe":
            self.model = TransE(
                node_num=node_num,
                relation_num=relaion_num,
                hidden_channels=hidden_channels,
                margin=margin
            )
        elif model_type == "transh":
            self.model = TransH(
                node_num=node_num,
                relation_num=relaion_num,
                hidden_channels=hidden_channels,
                margin=margin
            )
            
        elif model_type == "rotate":
            self.model = RotatE(
                node_num=node_num,
                relation_num=relaion_num,
                hidden_channels=hidden_channels,
                margin=margin
            )
        self.margin = margin
        self.p_norm = p_norm
        self.node_num = node_num
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
    
    def train(
        self,
        triplet: torch.tensor,
    ):
        self.model.train()
        total_loss = total_examples = 0
        loader = data_loader(triplet)
        
        for seed, (head_index, rel_type, tail_index) in enumerate(loader):
            self.optimizer.zero_grad()
            
            neg_head_index, neg_rel_type, neg_tail_index = random_sample(head_index, rel_type, tail_index, num_node=self.node_num, seed=seed)
            
            loss = self.model.loss(
                head_index.to(device), 
                rel_type.to(device), 
                tail_index.to(device), 
                neg_head_index.to(device), 
                neg_rel_type.to(device), 
                neg_tail_index.to(device),
            )
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * head_index.numel()
            total_examples += head_index.numel()
        
        return total_loss / total_examples
        
        
        
        
    
    @torch.no_grad()
    def valid(
        self,
        triplet: torch.tensor
    ):
        self.model.eval()
        loader = data_loader(triplet)
        total_loss = total_examples = 0
        for seed, (head_index, rel_type, tail_index) in enumerate(loader):
            neg_head_index, neg_rel_type, neg_tail_index = random_sample(head_index, rel_type, tail_index, self.node_num, seed)
            loss = self.model.loss(
                head_index.to(device), 
                rel_type.to(device), 
                tail_index.to(device), 
                neg_head_index.to(device), 
                neg_rel_type.to(device), 
                neg_tail_index.to(device),
            )
            total_loss += float(loss) * head_index.numel()
            total_examples += head_index.numel()
        return total_loss / total_examples
    
    @torch.no_grad()
    def test(
        self,
        test_triplet: torch.tensor,
        train_triples: torch.tensor,
        valid_triples: torch.tensor,
        batch_size: int,
        k: int = 10,
    ):
        head_index = test_triplet[:, 0]
        rel_type = test_triplet[:, 1]
        tail_index = test_triplet[:, 2]
        
        arange = range(head_index.numel())
        self.model.eval()
        exist_data = torch.cat([train_triples, valid_triples])
        mean_ranks, reciprocal_ranks, hits_at_k = [], [], []

        for i in arange:
            h, r, t = head_index[i], rel_type[i], tail_index[i]

            exist = exist_data[exist_data[:,0] == h.item()]
            exist_tail = exist[exist[:,1] == r.item()][:, 2]
            scores = []
            tail_indices = torch.arange(self.node_num)
            tail_indices = torch.tensor(list(set(tail_indices.tolist()) - set(exist_tail.tolist())))

            t = (tail_indices == t).nonzero(as_tuple=True)[0].to(device)


            for ts in tail_indices.split(batch_size):
                scores.append(self.model.forward(h.expand_as(ts).to(device), r.expand_as(ts).to(device), ts.to(device)))
            rank = int((torch.cat(scores).argsort(descending=True) == t).nonzero().view(-1)) + 1
            mean_ranks.append(rank)
            reciprocal_ranks.append(1 / (rank))
            hits_at_k.append(rank <= k)
        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)
        return mean_rank, mrr, hits_at_k
    
    def save_model(self, model_path: str = "./kge.pth"):
        torch.save(self.model.state_dict(), model_path)

    
    def run(
        self,
        train_triplet: torch.tensor,
        valid_triplet: torch.tensor,
        test_triplet: torch.tensor,
    ):
        for epoch in range(1, 101):
            train_loss = self.train(triplet=train_triplet)
            valid_loss = self.valid(triplet=valid_triplet)
            
            print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
            if epoch % 10 == 0:
                k = 10
                mean_rank, mrr, hits_at_k = self.test(
                    test_triplet=test_triplet,
                    valid_triples=valid_triplet,
                    train_triples=train_triplet,
                    batch_size=64,
                    k=k
                )
                
                print("##############")
                print(f"Mean Rank: {mean_rank:.4f}, MRR: {mrr:.4f}, Hits@{k}: {hits_at_k:.4f}")
    
    
    def get_embedding(self):
        model_path = './kge.pth'
        self.model.load_state_dict(torch.load(model_path))
        
        # print(self.model.node_emb.weight.size())
        
            

if __name__ == "__main__":
    triplet = pd.read_csv(f"{root_path}/data/processed/triplet.csv")[:722] #3299
    dataset = pd.read_csv(f"{root_path}/data/processed/dataset.csv")[:559] #1497
    triplet = change_index(data=dataset, triplets_df=triplet)
    
    train, valid, test = split_data(triplets=triplet)
    
    exe = Exec(
        model_type="rotate",
        node_num = len(dataset),
        relaion_num=len(relations),
        hidden_channels=256,
        p_norm=2.0,
        margin=2.0
    )
    
    # exe.get_embedding()
    
    exe.run(
        train_triplet=train,
        valid_triplet=valid,
        test_triplet=test
    )
    
    exe.save_model()
    