import sys # noqa
sys.path.append("../../") # noqa
import os
import torch
import pandas as pd
from typing import Tuple
from src.utils.hop_triplet_loader import HopTripletLoader
from model import BaseLineModel2
from src.utils.static_seed import static_seed
from sentence_bert import SentenceBert
from typing import Dict, Any
import torch.nn.functional as F
import tqdm

root_path = os.path.join(os.path.dirname(__file__), "../../")
static_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
relations = [
        "ParentOf", 
        "ChildOf", 
        "CanPrecede", 
        "CanFollow", 
        "PeerOf", 
        "TargetOf", 
        "AttackOf", 
        "InstanceOf", 
        "AbstractionOf"
    ]

raw_triplet = pd.read_csv(f"{root_path}/data/processed/triplet.csv")
raw_dataset = pd.read_csv(f"{root_path}/data/processed/dataset.csv")

node_num = len(raw_dataset)
rel_num = len(relations)

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
        if relation == 0 or relation == 2 or relation == 5:
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
    train_triplets_dict = {}
    valid_triplets_dict = {}
    test_triplets_dict = {}
    
    triplets = reverse_triplet(triplets=triplets)
    triple_num = len(triplets)
    rnd_index = torch.randperm(triple_num)
    triplets = triplets[rnd_index]
    
    train_index = int(triple_num * train)
    valid_index = int(triple_num * valid)
    train_triples = triplets[:train_index]
    valid_triples = triplets[train_index:train_index+valid_index]
    test_triples = triplets[train_index+valid_index:]
    
    train_triplets_dict["node_id"] = torch.arange(len(raw_dataset))
    valid_triplets_dict["node_id"] = torch.arange(len(raw_dataset))
    test_triplets_dict["node_id"] = torch.arange(len(raw_dataset))
    
    train_triplets_dict["edge_label_index"] = train_triples
    valid_triplets_dict["edge_label_index"] = valid_triples
    test_triplets_dict["edge_label_index"] = test_triples
    
    train_triplets_dict["edge_index"] = train_triples
    valid_triplets_dict["edge_index"] = train_triples
    test_triplets_dict["edge_index"] = torch.concat([train_triples, valid_triples])
    
    return train_triplets_dict, valid_triplets_dict, test_triplets_dict

def triplet_loader(triplet: Dict[str, torch.tensor]) -> HopTripletLoader:
    
    loader = HopTripletLoader(
        triplet_dict = triplet,
        neighbor_hop=2,
        add_negative_label=True,
        num_node=node_num,
        batch_size=32,
        shuffle=True,
    )
    
    return loader

def train(
    model: BaseLineModel2,
    loader: HopTripletLoader,
    encoder_optimizer: Any,
    decoder_optimizer: Any
):
    model.train()
    total_loss = total_examples = 0
    for data in tqdm.tqdm(loader):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        edge_index = data["edge_index"].to(device)
        edge_label_index = data["edge_label_index"].to(device)
        node_id = data["node_id"].to(device)
        positive_edge_index = data["positive"].to(device)
        negative_edge_index = data["negative"].to(device)
        
        pos_encoder_score, pos_decoder_score, pos_l2 = model.forward(
            node_id=node_id,
            head_index=edge_index[:, 0],
            rel_type=edge_index[:, 1],
            tail_index=edge_index[:, 2],
            head_label_index=positive_edge_index[:, 0],
            rel_label_type=positive_edge_index[:, 1],
            tail_label_index=positive_edge_index[:, 2]
        )

        neg_encoder_score, neg_decoder_score, neg_l2 = model.forward(
            node_id=node_id,
            head_index=edge_index[:, 0],
            rel_type=edge_index[:, 1],
            tail_index=edge_index[:, 2],
            head_label_index=negative_edge_index[:, 0],
            rel_label_type=negative_edge_index[:, 1],
            tail_label_index=negative_edge_index[:, 2]
        )
        
        encoder_loss = F.margin_ranking_loss(
            pos_encoder_score,
            neg_encoder_score,
            target=torch.ones_like(pos_encoder_score),
            margin=2.0,
        )
        
        decoder_score = torch.cat([pos_decoder_score, neg_decoder_score])
        y = torch.cat([torch.ones_like(pos_decoder_score), -1 * torch.ones_like(neg_decoder_score)]) 
        l2 = (pos_l2 + neg_l2) / 2
        
        decoder_loss = torch.mean(F.softplus(decoder_score * y))
        
        encoder_loss.backward(retain_graph=True)
        decoder_loss.backward(retain_graph=True)
        encoder_optimizer.step()  
        decoder_optimizer.step()   
    
    return encoder_loss, decoder_loss


@torch.no_grad()
def valid(
    model: BaseLineModel2,
    loader: HopTripletLoader
):
    model.eval()
    
    return 1


@torch.no_grad()
def test(
    model: BaseLineModel2,
    loader: HopTripletLoader
):
    model.eval()

def main():    
    triplet = change_index(data=raw_dataset, triplets_df=raw_triplet)
    train_triples, valid_triples, test_triples = split_data(triplets=triplet)
    
    train_loader = triplet_loader(triplet=train_triples)
    valid_loader = triplet_loader(triplet=valid_triples)
    
    # sentence_bert = SentenceBert()
    # embs = []
    # for series in raw_dataset.to_dict(orient="records"):
    #     name = series["Name"] if isinstance(series["Name"], str) else ""
    #     desciption = series["Description"] if isinstance(series["Description"], str) else ""
    #     sentence = name + " " + desciption
    #     emb = sentence_bert(sentence=sentence)
    #     embs.append(emb)
        
    # text_embedding = torch.cat(embs, dim=0)
    
    model = BaseLineModel2(
        structure_node_embedding=torch.randn(node_num, 128).to(device),
        text_node_embedding=torch.randn(node_num, 768).to(device),
        node_num=node_num,
        rel_num=rel_num
    ).to(device)

    encoder_optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=1e-3)
    decoder_optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-3)
    
    for epoch in range(1, 501):
        train_encoder_loss, train_decoder_loss = train(
            loader=train_loader, 
            model=model,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer
        )
        valid_loss = valid(
            model=model,
            loader=valid_loader
        )
        print(f'Epoch: {epoch:03d}, Train Encoder Loss: {train_encoder_loss:.4f}, Train Decoder Loss: {train_decoder_loss:.4f}, Valid_Loss: {valid_loss:.4f}')
        
        # if epoch % 30 == 0:
        #     index = (test_triples[:, 1] == 1 ).nonzero(as_tuple=True)[0]
        #     mean_rank, mrr, hits_at_k = test(
        #         model=model,
        #         head_index=test_triples[:, 0],
        #         rel_type=test_triples[:, 1],
        #         tail_index=test_triples[:, 2],
        #         train_triples=train_triples,
        #         valid_triples=valid_triples,
        #         batch_size=64
        #     )
        #     print(f'MeanRank: {mean_rank:.4f}, MRR: {mrr:.4f}, Hits@10: {hits_at_k:.4f}')
    
    
main()