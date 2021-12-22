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

        # shape : (batch, length, num_labels)
        logits = self.score(last_hidden_state)

        # get the index of the final representation
        sequence_lengths = torch.ne(attention_mask, 0).sum(-1) - 1

        # shape : (batch, num_labels)
        pooled_logits = logits[range(batch_size), sequence_lengths]

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
                loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(pooled_logits, labels)
        elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(pooled_logits, labels)
        ## until here ##

        # shape : (batch, )
        predictions = pooled_logits.argmax(dim=-1)

        # loss        : (batch, )
        # predictions : (batch, )
        return loss, predictions


class RobertaOutputProcessor(BaseOutputProcessor):
    def __init__(self, config, embedding_dim, num_labels):
        super().__init__(config, embedding_dim, num_labels)

        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)

    def forward(
        self,
        last_hidden_state:torch.Tensor,                # shape : (batch, length, embedding_dim)
        attention_mask:torch.Tensor,        # shape : (batch, length)
        labels:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, _ = attention_mask.shape
        pooled_hidden_state = last_hidden_state[:,0,:]
        pooled_hidden_state = self.dropout(pooled_hidden_state)
        pooled_hidden_state = self.dense(pooled_hidden_state)
        pooled_hidden_state = torch.tanh(pooled_hidden_state)
        pooled_hidden_state = self.dropout(pooled_hidden_state)
        
        # shape : (batch, num_labels)
        pooled_logits = self.score(pooled_hidden_state)

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
                loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(pooled_logits, labels)
        elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(pooled_logits, labels)
        ## until here ##

        # shape : (batch, )
        predictions = pooled_logits.argmax(dim=-1)

        # loss        : (batch, )
        # predictions : (batch, )
        return loss, predictions