import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def entity_rank(
        model: any,
        head_index: torch.Tensor,
        rel_type: torch.Tensor,
        tail_index: torch.Tensor,
        train_triples,
        valid_triples,
        batch_size: int,
        k: int = 5,
):
        arange = range(head_index.numel())
        model.eval()
        
        exist_data = torch.cat([train_triples, valid_triples])
        mean_ranks, reciprocal_ranks, hits_at_k = [], [], []
        
        for i in arange:
            h, r, t = head_index[i], rel_type[i], tail_index[i]
            
            exist = exist_data[exist_data[:,0] == h.item()]
            exist_tail = exist[exist[:,1] == r.item()][:, 2]
            scores = []
            tail_indices = torch.arange(1497)
            tail_indices = torch.tensor(list(set(tail_indices.tolist()) - set(exist_tail.tolist())))
            t = (tail_indices == t).nonzero(as_tuple=True)[0].to(device)
            
            for ts in tail_indices.split(batch_size):
                scores.append(model.forward(h.expand_as(ts).to(device), r.expand_as(ts).to(device), ts.to(device)))
            rank = int((torch.cat(scores).argsort(
                descending=True) == t).nonzero().view(-1)) + 1
            mean_ranks.append(rank)
            reciprocal_ranks.append(1 / (rank))
            hits_at_k.append(rank <= k)

        mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
        mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
        hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)

        return mean_rank, mrr, hits_at_k