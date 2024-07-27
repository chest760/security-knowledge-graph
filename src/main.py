import os
import torch
import pandas as pd
from typing import Tuple
from data import CreateHeteroData
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from model import Model
from models.gnn_models.hgt import HGT
from models.gnn_models.gat import GAT
from models.gnn_models.rgat import RGAT
from utils.static_seed import static_seed

static_seed(42)

root_path = os.path.join(os.path.dirname(__file__), "../")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def graph_loader(data: HeteroData, data_type: str = "train"):
    loader = LinkNeighborLoader(
        data,
        edge_label_index=[("capec", "to", "capec"), data['capec', 'to', 'capec'].edge_label_index],
        edge_label=data['capec', 'to', 'capec'].edge_label,
        neg_sampling_ratio=0.0,
        num_neighbors=[-1] * 2,
        batch_size=32,
        shuffle=True if data_type == "train" else False,
    )
    
    return loader


class Execute(torch.nn.Module):
    def __init__(
        self,
        train_graph: HeteroData,
        valid_graph: HeteroData,
        test_graph: HeteroData
    ):
        super().__init__()
        self.train_graph = train_graph
        self.valid_graph = valid_graph
        self.test_graph = test_graph

        gnn_model = HGT(
                hidden_channels=1024*2 + 256,
                out_channels=256,
                data=train_graph
            ).to(device)
        
        gnn_model = GAT().to(device)
        
        # gnn_model = RGAT(
        #     in_channels=1024*2 + 256,
        #     out_channels=256,
        #     num_relations=5,
        #     heads=1
        # ).to(device)
        
        self.model: Model = Model(
            gnn_model=gnn_model
        ).to(device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
    
    def init_data(self):
        pass
        
    
    def train(self):
        self.model.train()
        total_loss = total_examples = 0
        loader = graph_loader(self.train_graph, "train")
        for seed, data in enumerate(loader):
            self.optimizer.zero_grad()
            data.to(device)
            neg_head, rel_type, neg_tail = random_sample(
                head_index=data["capec", "to", "capec"].edge_label_index[0],
                rel_type=data["capec", "to", "capec"].edge_label,
                tail_index=data["capec", "to", "capec"].edge_label_index[1],
                num_node=len(data["capec"].n_id),
                seed=seed
            )
            neg_edge_label_index = torch.stack([neg_head, neg_tail])
  
            
            edge_label = self.train_graph["capec", "to", "capec"].edge_label[data["capec", "to", "capec"].e_id.cpu()].to(device)
            
            loss = self.model.loss(
                x_dict=data.x_dict,
                edge_index_dict=data.edge_index_dict,
                edge_label=edge_label,
                pos_edge_label=data["capec", "to", "capec"].edge_label,
                pos_edge_label_index=data["capec", "to", "capec"].edge_label_index,
                neg_edge_label_index=neg_edge_label_index,
                neg_edge_label=rel_type
            )
            
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * neg_head.numel()
            total_examples += neg_head.numel()
        return total_loss / total_examples
            
            
    
    @torch.no_grad()
    def valid(self):
        self.model.eval()
        total_loss = total_examples = 0
        loader = graph_loader(self.valid_graph, "test")
        for seed, data in enumerate(loader):
            data.to(device)
            neg_head, rel_type, neg_tail = random_sample(
                head_index=data["capec", "to", "capec"].edge_label_index[0],
                rel_type=data["capec", "to", "capec"].edge_label,
                tail_index=data["capec", "to", "capec"].edge_label_index[1],
                num_node=len(data["capec"].n_id),
                seed=seed
            )
            neg_edge_label_index = torch.stack([neg_head, neg_tail])
            
            edge_label = self.train_graph["capec", "to", "capec"].edge_label[data["capec", "to", "capec"].e_id.cpu()].to(device)
            
            loss = self.model.loss(
                x_dict=data.x_dict,
                edge_index_dict=data.edge_index_dict,
                edge_label=edge_label,
                pos_edge_label=data["capec", "to", "capec"].edge_label,
                pos_edge_label_index=data["capec", "to", "capec"].edge_label_index,
                neg_edge_label_index=neg_edge_label_index,
                neg_edge_label=rel_type
            )
            
            total_loss += float(loss) * neg_head.numel()
            total_examples += neg_head.numel()
        return total_loss / total_examples

    @torch.no_grad()
    def test(
        self,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
        exist_triplet: torch.tensor,
        batch_size: int,
        k: int = 10,
    ):
        arange = range(head_index.numel())
        self.model.eval()
        
        self.test_graph.to(device)

        mean_ranks, reciprocal_ranks, hits_at_k = [], [], []
        
        edge_label = torch.concat([self.train_graph["capec", "to", "capec"].edge_label, self.valid_graph["capec", "to", "capec"].edge_label]).to(device)
        
        x_dict = self.model.forward(
            self.test_graph.x_dict,
            self.test_graph.edge_index_dict,
            edge_label=edge_label,
        )
        
        x = x_dict["capec"]

        for i in arange:
            h, r, t = head_index[i], rel_type[i], tail_index[i]

            scores = []
            exist = exist_triplet[exist_triplet[:,0] == h.item()]
            exist_tail = exist[exist[:,1] == r.item()][:, 2].to(device)
            tail_indices = torch.arange(559).to(device)
            tail_indices = torch.tensor(list(set(tail_indices.tolist()) - set(exist_tail.tolist()))).to(device)

            t = (tail_indices == t).nonzero(as_tuple=True)[0].to(device)

            for ts in tail_indices.split(batch_size):
                score = self.model._calc_score(x=x, edge_label_index=torch.stack([h.expand_as(ts).to(device), ts.to(device)]), edge_label=r.expand_as(ts).to(device))
                scores.append(score)
            rank = int((torch.cat(scores).argsort(descending=True) == t).nonzero().view(-1)) + 1
            mean_ranks.append(rank)
            reciprocal_ranks.append(1 / (rank))
            hits_at_k.append(rank <= k)
        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)
        return mean_rank, mrr, hits_at_k
    
    
    def run(self):
        train_valid_edge_label_index = torch.concat([self.train_graph["capec", "to", "capec"].edge_label_index, self.valid_graph["capec", "to", "capec"].edge_label_index], dim=1)
        train_valid_edge_label = torch.concat([self.train_graph["capec", "to", "capec"].edge_label, self.valid_graph["capec", "to", "capec"].edge_label], dim=0)
        
        exist_triplet = torch.stack([
            train_valid_edge_label_index[0],
            train_valid_edge_label,
            train_valid_edge_label_index[1]
        ]).transpose(0,1)
    
        
        for epoch in range(1,101):
            train_loss = self.train()
            valid_loss = self.valid()
            print(f"Epoch: {epoch}, Train Loss: {train_loss}, Valid loss: {valid_loss}")
            
            if epoch % 10 == 0:
                k = 10
                mean_rank, mrr, hits_k = self.test(
                    head_index=self.test_graph["capec", "to", "capec"].edge_label_index[0],
                    rel_type=self.test_graph["capec", "to", "capec"].edge_label,
                    tail_index=self.test_graph["capec", "to", "capec"].edge_label_index[1],
                    exist_triplet=exist_triplet,
                    batch_size=32,
                    k=k
                )
                print("\n###########################")
                print(f"Mean Rank: {mean_rank}, MRR: {mrr}, Hits@{k}: {hits_k}") 
                print("###########################\n")
            


def main():
    triplet = pd.read_csv(f"{root_path}/data/processed/triplet.csv")[:722]
    dataset = pd.read_csv(f"{root_path}/data/processed/dataset.csv")[:559]
    
    train_graph, valid_graph, test_graph = CreateHeteroData(
        triplet=triplet,
        dataset=dataset,
        text_embedding_model="voyage",
        graph_embedding_model="rotate"
    ).init_graph()
    
    exec = Execute(
        train_graph=train_graph,
        valid_graph=valid_graph,
        test_graph=test_graph
    )
    
    exec.run()
    
    
if __name__ == "__main__":
    main()