
from typing import Tuple
import time

import torch

from transformers import AutoModel

from .InputProcessor import ReverseInputProcessor, BaseInputProcessor
from .OutputProcessor import BaseOutputProcessor, ReverseInputOutputProcessor



class GPT2Wrapper(torch.nn.Module):
    def __init__(self, config, model_name_or_path):
        super(GPT2Wrapper, self).__init__()

        self.config = config

        # Main model
        self.transformer = AutoModel.from_pretrained(
                                        model_name_or_path,
                                        from_tf=bool(".ckpt" in model_name_or_path),
                                        config=config)

        self.embedding_dim = self.transformer.wte.embedding_dim
        self.num_labels = config.num_labels

        if config.apply_reverse:
            print('REVERSE INPUT AND OUTPUT....')
            time.sleep(3)
            self.input_processor = ReverseInputProcessor(config=config, embeddings=self.transformer.wte)
            # goes through (embedding_dim * 2, num_label) linear layer
            self.output_processor = ReverseInputOutputProcessor(config=config, embedding_dim=self.embedding_dim, num_labels=self.num_labels)
        else:
            print('ORIGINAL INPUT AND OUTPUT....')
            time.sleep(3)
            # default input and output processor for out toy task
            self.input_processor = BaseInputProcessor(config=config, embeddings=self.transformer.wte)
            # goes through (embedding_dim, num_label) linear layer
            self.output_processor = BaseOutputProcessor(config=config, embedding_dim=self.embedding_dim, num_labels=self.num_labels)
        
    def forward(
        self,
        input_ids=None,
        reversed_input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # inputs_embeds  : (batch, input_length, embedding_dim)
        # attention_mask : (batch, input_length)
        if self.config.apply_reverse:
            inputs_embeds, reversed_input_embeddings, attention_mask = self.input_processor(input_ids=input_ids, reversed_input_ids=reversed_input_ids, attention_mask=attention_mask)
            outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            # reversed_input
            reversed_outputs = self.transformer(inputs_embeds=reversed_input_embeddings, attention_mask=attention_mask)
            
            
            # # shape : (batch, length, embedding_dim)
            last_hidden_state = outputs.last_hidden_state
            # # shape : (batch, length, embedding_dim)
            reversed_last_hidden_state = reversed_outputs.last_hidden_state

            # shape : (batch, length, embedding_dim * 2)
            last_hidden_state = torch.cat([last_hidden_state, reversed_last_hidden_state], dim=2)
        else:
            inputs_embeds, attention_mask = self.input_processor(input_ids=input_ids, attention_mask=attention_mask)
            outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

            # shape : (batch, length, embedding_dim)
            last_hidden_state = outputs.last_hidden_state
        # loss        : (batch, )
        # predictions : (batch, )
        loss, predictions = self.output_processor(last_hidden_state=last_hidden_state, attention_mask=attention_mask, labels=labels)

        return loss, predictions
