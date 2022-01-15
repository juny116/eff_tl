

from typing import Tuple

import torch
from transformers import AutoModel, AutoConfig

class BaseInputProcessor(torch.nn.Module):
    def __init__(self, config, embeddings):
        super(BaseInputProcessor, self).__init__()
        
        self.config = config
        self.embeddings = embeddings
        self.embedding_dim = self.embeddings.embedding_dim

    def forward(
        self,
        input_ids:torch.Tensor, 
        attention_mask:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, length = input_ids.shape

        # shape : (batch, length, embedding_dim)
        input_embeddings = self.embeddings(input_ids)

        assert batch_size == input_embeddings.shape[0]
        assert length == input_embeddings.shape[1]

        # input_embeddings : (batch, length, embedding_dim)
        # attention_mask   : (batch, length)
        return input_embeddings, attention_mask


class ReverseInputProcessor(BaseInputProcessor):
    def __init__(self, config, embeddings):
        super().__init__(config, embeddings)
        

    def forward(
        self,
        input_ids:torch.Tensor, 
        reversed_input_ids:torch.Tensor,
        attention_mask:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, length = input_ids.shape

        # shape : (batch, length, embedding_dim)
        input_embeddings = self.embeddings(input_ids)
        reversed_input_embeddings = self.embeddings(reversed_input_ids)

        assert batch_size == input_embeddings.shape[0]
        assert length == input_embeddings.shape[1]
        assert batch_size == reversed_input_embeddings.shape[0]
        assert length == reversed_input_embeddings.shape[1]

        # input_embeddings : (batch, length, embedding_dim)
        # attention_mask   : (batch, length)
        return input_embeddings, reversed_input_embeddings, attention_mask