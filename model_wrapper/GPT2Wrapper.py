
from typing import Tuple

import torch

from transformers import AutoModel
from transformers import GPT2Tokenizer

from .InputProcessor import BaseInputProcessor
from .OutputProcessor import BaseOutputProcessor, MLMOutputProcessor



class GPT2Wrapper(torch.nn.Module):
    def __init__(self, config, model_name_or_path):
        super(GPT2Wrapper, self).__init__()

        self.config = config

        # Main model
        self.transformer = AutoModel.from_pretrained(model_name_or_path,
                                                    from_tf=bool(".ckpt" in model_name_or_path),
                                                    config=config)

        self.embedding_dim = self.transformer.wte.embedding_dim
        self.num_labels = config.num_labels

        # for output processing (output logits -> loss, prediction)
        self._output_processor = BaseOutputProcessor(config=config, embedding_dim=self.embedding_dim, num_labels=self.num_labels)
        self.mlm_output_processor = MLMOutputProcessor(config=config, embedding_dim=self.embedding_dim)
        
        self.input_processor = BaseInputProcessor(config=config, embeddings=self.transformer.wte)
               

    def forward(
        self,
        input_ids=None,
        mlm_input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        mlm_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # inputs_embeds  : (batch, input_length, embedding_dim)
        # attention_mask : (batch, input_length)
        inputs_embeds, mlm_inputs_embeds, attention_mask = self.input_processor(input_ids=input_ids, mlm_input_ids=mlm_input_ids, attention_mask=attention_mask)

        # for classification
        outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        # shape : (batch, length, embedding_dim)
        last_hidden_state = outputs.last_hidden_state
        loss, predictions = self._output_processor(last_hidden_state=last_hidden_state, attention_mask=attention_mask, labels=labels)
        return loss, predictions

        # for mlm task
        # mlm_outputs = self.transformer(inputs_embeds=mlm_inputs_embeds, attention_mask=attention_mask)
        # # shape : (batch, length, embedding_dim)
        # mlm_last_hidden_state = mlm_outputs.last_hidden_state
        # mlm_loss = self.mlm_output_processor(last_hidden_state=mlm_last_hidden_state, attention_mask=attention_mask, labels=mlm_labels)
        
        # return mlm_loss

        #return loss, mlm_loss, predictions
