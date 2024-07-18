import sys # noqa
sys.path.append("../../") # noqa
import os
import tqdm
import torch
import pandas as pd
from typing import Tuple
from encoder import Encoder
from decoder import Decoder
from typing import Dict, Any
import torch.nn.functional as F
from sentence_bert import SentenceBert
from src.utils.static_seed import static_seed
from src.utils.hop_triplet_loader import HopTripletLoader

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

raw_triplet = pd.read_csv(f"{root_path}/data/processed/triplet.csv")[:722]
raw_dataset = pd.read_csv(f"{root_path}/data/processed/dataset.csv")[:559]

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
    loader: HopTripletLoader,
    encoder: Encoder,
    decoder: Decoder,
    encoder_optimizer: Any,
    decoder_optimizer: Any,
):
    encoder.train()
    decoder.train()
    total_encoder_loss = total_examples = 0
    total_decoder_loss = 0
    for data in tqdm.tqdm(loader):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        edge_index = data["edge_index"].to(device)
        edge_label_index = data["edge_label_index"].to(device)
        node_id = data["node_id"].to(device)
        positive_edge_index = data["positive"].to(device)
        negative_edge_index = data["negative"].to(device)
        
        encoder_loss, x, rel_emb = encoder.loss(
            node_id=node_id,
            head_index=edge_index[:, 0],
            rel_type=edge_index[:, 1],
            tail_index=edge_index[:, 2],
            pos_head_index=positive_edge_index[:, 0],
            pos_rel_type=positive_edge_index[:, 1],
            pos_tail_index=positive_edge_index[:, 2],
            neg_head_index=negative_edge_index[:, 0],
            neg_rel_type=negative_edge_index[:, 1],
            neg_tail_index=negative_edge_index[:, 2]
        )
        
        decoder_loss = decoder.loss(
            x=x,
            rel_emb=rel_emb,
            pos_head_index=positive_edge_index[:, 0],
            pos_rel_type=positive_edge_index[:, 1],
            pos_tail_index=positive_edge_index[:, 2],
            neg_head_index=negative_edge_index[:, 0],
            neg_rel_type=negative_edge_index[:, 1],
            neg_tail_index=negative_edge_index[:, 2]
        )
        
        
        encoder_loss.backward(retain_graph=True)
        decoder_loss.backward(retain_graph=True)
        encoder_optimizer.step()  
        decoder_optimizer.step()
        
        total_encoder_loss += float(encoder_loss) * positive_edge_index[:, 0].numel()
        total_decoder_loss += float(decoder_loss) * positive_edge_index[:, 0].numel()
        total_examples += positive_edge_index[:, 0].numel()
    
    return total_encoder_loss/total_examples, total_decoder_loss/total_examples


@torch.no_grad()
def valid(
    loader: HopTripletLoader,
    encoder: Encoder,
    decoder: Decoder,
):
    encoder.eval()
    decoder.eval()
    total_encoder_loss = total_examples = 0
    total_decoder_loss = 0
    for data in loader:        
        edge_index = data["edge_index"].to(device)
        edge_label_index = data["edge_label_index"].to(device)
        node_id = data["node_id"].to(device)
        positive_edge_index = data["positive"].to(device)
        negative_edge_index = data["negative"].to(device)
        
        encoder_loss, x, rel_emb = encoder.loss(
            node_id=node_id,
            head_index=edge_index[:, 0],
            rel_type=edge_index[:, 1],
            tail_index=edge_index[:, 2],
            pos_head_index=positive_edge_index[:, 0],
            pos_rel_type=positive_edge_index[:, 1],
            pos_tail_index=positive_edge_index[:, 2],
            neg_head_index=negative_edge_index[:, 0],
            neg_rel_type=negative_edge_index[:, 1],
            neg_tail_index=negative_edge_index[:, 2]
        )
        
        decoder_loss = decoder.loss(
            x=x,
            rel_emb=rel_emb,
            pos_head_index=positive_edge_index[:, 0],
            pos_rel_type=positive_edge_index[:, 1],
            pos_tail_index=positive_edge_index[:, 2],
            neg_head_index=negative_edge_index[:, 0],
            neg_rel_type=negative_edge_index[:, 1],
            neg_tail_index=negative_edge_index[:, 2]
        )
        
        total_encoder_loss += float(encoder_loss) * positive_edge_index[:, 0].numel()
        total_decoder_loss += float(decoder_loss) * positive_edge_index[:, 0].numel()
        total_examples += positive_edge_index[:, 0].numel()
    
    return total_encoder_loss/total_examples, total_decoder_loss/total_examples


@torch.no_grad()
def test(
    encoder: Encoder,
    decoder: Decoder,
    head_index: torch.Tensor,
    rel_type: torch.Tensor,
    tail_index: torch.Tensor,
    train_triples,
    valid_triples,
    batch_size: int,
    k: int = 10,
):
    arange = range(head_index.numel())
    encoder.eval()
    decoder.eval()
    
    exist_data = torch.cat([train_triples, valid_triples])
    mean_ranks, reciprocal_ranks, hits_at_k = [], [], []
    
    for i in arange:
        h, r, t = head_index[i], rel_type[i], tail_index[i]
        
        exist = exist_data[exist_data[:,0] == h.item()]
        exist_tail = exist[exist[:,1] == r.item()][:, 2]
        scores = []
        tail_indices = torch.arange(len(raw_dataset))
        # tail_indices = torch.tensor(list(set(tail_indices.tolist()) - set(exist_tail.tolist())))
        
        # t = (tail_indices == t).nonzero(as_tuple=True)[0].to(device)
        
        node_id = torch.arange(len(raw_dataset))
                
        for ts in tail_indices.split(batch_size):
            x, rel_embedding = encoder.forward(
                node_id,
                h.expand_as(ts).to(device), 
                r.expand_as(ts).to(device), 
                ts.to(device)
            )
            
            score = encoder._calc_score(
                x,
                h.expand_as(ts).to(device), 
                r.expand_as(ts).to(device), 
                ts.to(device)
            )
        
            # score, _ = decoder.forward(
            #     x=x,
            #     rel_emb=rel_embedding,
            #     head_index=h.expand_as(ts).to(device), 
            #     rel_type=r.expand_as(ts).to(device), 
            #     tail_index=ts.to(device)
            # )
            
        
            scores.append(score)
        rank = int((torch.cat(scores).argsort(descending=True) == t).nonzero().view(-1)) + 1
        mean_ranks.append(rank)
        reciprocal_ranks.append(1 / (rank))
        hits_at_k.append(rank <= k)
    mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
    mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
    hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)
    return mean_rank, mrr, hits_at_k

@torch.no_grad()
def get_text_embedding():
    sentence_bert = SentenceBert().to(device)
    embs = []
    for series in raw_dataset.to_dict(orient="records"):
        name = series["Name"] if isinstance(series["Name"], str) else ""
        desciption = series["Description"] if isinstance(series["Description"], str) else ""
        sentence = name + " " + desciption
        emb = sentence_bert.forward(sentence=sentence)
        embs.append(emb)
    text_embedding = torch.cat(embs, dim=0)
    
    return text_embedding

def main():    
    triplet = change_index(data=raw_dataset, triplets_df=raw_triplet)
    train_triples, valid_triples, test_triples = split_data(triplets=triplet)
    
    train_loader = triplet_loader(triplet=train_triples)
    valid_loader = triplet_loader(triplet=valid_triples)
    
    
    text_embedding = get_text_embedding()
    
    encoder = Encoder(
        text_embedding=text_embedding.to(device),
        structure_embedding=torch.randn(node_num, 128).to(device),
        node_num=node_num,
        rel_num=rel_num
    ).to(device)

    decoder = Decoder(
        node_num=node_num,
        rel_num=rel_num,
        kernel_size=1,
        hidden_channels=128,
        out_channels=128
    ).to(device)
    
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=2e-4)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=2e-4)
    
    for epoch in range(1, 721):
        train_encoder_loss, train_decoder_loss = train(
            loader=train_loader, 
            encoder=encoder,
            decoder=decoder,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
        )
        valid_encoder_loss, valid_decoder_loss = valid(
            loader=valid_loader,
            encoder=encoder,
            decoder=decoder,
        )
        print(f'Epoch: {epoch:03d}, Train Encoder Loss: {train_encoder_loss:.4f}, Train Decoder Loss: {train_decoder_loss:.4f}, Valid Encoder Loss: {valid_encoder_loss:.4f}, Valid Decoder Loss: {valid_decoder_loss:.4f},')
        
        if epoch % 20 == 0:
            mean_rank, mrr, hits_at_k = test(
                encoder=encoder,
                decoder=decoder,
                head_index=test_triples["edge_label_index"][:, 0],
                rel_type=test_triples["edge_label_index"][:, 1],
                tail_index=test_triples["edge_label_index"][:, 2],
                train_triples=train_triples["edge_label_index"],
                valid_triples=valid_triples["edge_label_index"],
                batch_size=512
            )
            print(f'MeanRank: {mean_rank:.4f}, MRR: {mrr:.4f}, Hits@10: {hits_at_k:.4f}')
    
    
main()