
from typing import Tuple

import torch

from transformers import AutoModel

from .RobertaInputProcessor import *
from .RobertaOutputProcessor import BaseOutputProcessor



class RobertaWrapper(torch.nn.Module):
    def __init__(self, config, model_name_or_path):
        super(RobertaWrapper, self).__init__()

        self.config = config

        # Main model
        self.transformer = AutoModel.from_pretrained(
                                        model_name_or_path,
                                        from_tf=bool(".ckpt" in model_name_or_path),
                                        config=config)

        self.embedding_dim = config.hidden_size
        self.num_labels = config.num_labels

        # for output processing (output logits -> loss, prediction)
        self.output_processor = BaseOutputProcessor(config=config, embedding_dim=self.embedding_dim, num_labels=self.num_labels)

        # # for other methods (LoRA, Adapter, Prefix-tuning)
        # input_ids -> input_embeds
        if not self.config.apply_input and not self.config.apply_encoder and self.config.prompt_length is None:
            self.input_processor = BaseInputProcessor(config=config, embeddings=self.transformer.get_input_embeddings())
        # for PROMPT_TUNING
        elif not self.config.apply_input and not self.config.apply_encoder:
            self.input_processor = PromptInputProcessor(config=config, embeddings=self.transformer.get_input_embeddings())
        # for PLM encoder + prompt only
        elif not self.config.apply_input and self.config.apply_encoder:
            self.input_processor = PromptEncoderInputProcessor(config=config, embeddings=self.transformer.get_input_embeddings())
        # for PLM encoder + input dependent
        elif self.config.apply_input and self.config.apply_encoder:
            self.input_processor = EncoderInputProcessor(config=config, embeddings=self.transformer.get_input_embeddings())
        

    def forward(
        self,
        input_ids=None,
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
        inputs_embeds, attention_mask = self.input_processor(input_ids=input_ids, attention_mask=attention_mask)

        outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        # shape : (batch, length, embedding_dim)
        last_hidden_state = outputs.last_hidden_state

        # loss        : (batch, )
        # predictions : (batch, )
        loss, predictions = self.output_processor(last_hidden_state=last_hidden_state, attention_mask=attention_mask, labels=labels)

        return loss, predictions
