from typing import Tuple

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class BaseOutputProcessor(torch.nn.Module):
    def __init__(self, config, embedding_dim, num_labels):
        super(BaseOutputProcessor, self).__init__()

        self.config = config
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels

        # final layer for prediction
        self.score = torch.nn.Linear(self.embedding_dim, self.num_labels, bias=False)
        # self.score.to(dtype=torch.half, non_blocking=True)

    def forward(
        self,
        last_hidden_state:torch.Tensor,                # shape : (batch, length, embedding_dim)
        attention_mask:torch.Tensor,        # shape : (batch, length)
        labels:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, _ = attention_mask.shape

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # shape : (batch, embedding_dim)
        mean_embedding = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # shape : (batch, num_labels)
        logits = self.score(mean_embedding.half())

        # # for last hidden state representation
        # sequence_lengths = torch.ne(attention_mask, 0).sum(-1) -1
        # # shape : (batch, length, num_labels)
        # full_logits = self.score(last_hidden_state)
        # logits = full_logits[range(batch_size), sequence_lengths]
        # # for last hidden state representation

        ## same code as transformers.GPT2ForSequenceClassification ##
        loss = None
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        if self.config.problem_type == "regression":
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        ## until here ##

        # shape : (batch, )
        predictions = logits.argmax(dim=-1)

        # loss        : (batch, )
        # predictions : (batch, )
        return loss, predictions


class MLMOutputProcessor(BaseOutputProcessor):
    def __init__(self, config, embedding_dim, num_labels):
        super().__init__(config, embedding_dim, num_labels)

        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(
        self,
        last_hidden_state:torch.Tensor,                # shape : (batch, length, embedding_dim)
        attention_mask:torch.Tensor,        # shape : (batch, length)
        labels:torch.Tensor,
    ) -> torch.Tensor:
        
        lm_logits = self.lm_head(last_hidden_state)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss
