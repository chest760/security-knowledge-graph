import torch
from .base import TextEmbedding
from transformers import RobertaModel, RobertaTokenizer


class SecureBERT(TextEmbedding):
    def __init__(self):
        super().__init__()
        self.model = RobertaModel.from_pretrained("ehsanaghaei/SecureBERT")
        self.tokenizer = RobertaTokenizer.from_pretrained("ehsanaghaei/SecureBERT")

    def forward(
        self,
        id: torch.LongTensor,
        mask: torch.LongTensor
    ):
        output = self.model(id, mask, output_hidden_states=True)
        return output.pooler_output
