

from typing import Tuple

import torch
from transformers import AutoModel

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
        # shape : (batch, length, embedding_dim)
        input_embeddings = self.embeddings(input_ids)

        # input_embeddings : (batch, length, embedding_dim)
        # attention_mask   : (batch, length)
        return input_embeddings, attention_mask


## for Prompt-tuning ##
class PromptInputProcessor(BaseInputProcessor):
    def __init__(self, config, embeddings):
        super().__init__(config, embeddings)

        self.encoder_prompt_length = self.config.prompt_length
        assert self.encoder_prompt_length > 0, f'Prompt length must be greater than 0, got {self.encoder_prompt_length}.'
        # shape : (prompt_length, )
        self.encoder_prompt_inputs = torch.LongTensor(list(range(self.encoder_prompt_length)))
        # shape : (prompt_length, embedding_dim)
        self.encoder_prompt_embeddings = torch.nn.Embedding(self.encoder_prompt_length, self.embedding_dim)
        # shape : (prompt_length, )
        self.encoder_prompt_attention_mask = torch.ones(self.encoder_prompt_length)
                
    def forward(
        self,
        input_ids:torch.Tensor, 
        attention_mask:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, _ = input_ids.shape

        # shape : (batch, length, embedding_dim)
        input_embeddings = self.embeddings(input_ids)

        # shape : (prompt_length, embedding_dim)
        prompt_embeddings = self.encoder_prompt_embeddings(self.encoder_prompt_inputs)
        # shape : (batch, prompt_length, embedding_dim)
        prompt_embeddings = prompt_embeddings.unsqueeze(0).expand(batch_size, self.encoder_prompt_length, -1)

        # shape : (batch, prompt_length)
        prompt_attention_mask = self.encoder_prompt_attention_mask.unsqueeze(0).expand(batch_size, self.encoder_prompt_length)

        input_embeddings = torch.cat([prompt_embeddings, input_embeddings], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)

        # input_embeddings : (batch, length, embedding_dim)
        # attention_mask   : (batch, length)
        return input_embeddings, attention_mask


## for Prompt+Encoder-tuning ##
class PromptEncoderInputProcessor(BaseInputProcessor):
    def __init__(self, config, embeddings):
        super().__init__(config, embeddings)

        self.encoder_prompt_length = self.config.prompt_length
        assert self.encoder_prompt_length > 0, f'Prompt length must be greater than 0, got {self.encoder_prompt_length}.'
        # shape : (prompt_length, )
        self.encoder_prompt_inputs = torch.LongTensor(list(range(self.encoder_prompt_length)))
        # shape : (prompt_length, embedding_dim)
        self.encoder_prompt_embeddings = torch.nn.Embedding(self.encoder_prompt_length, self.embedding_dim)
        # shape : (prompt_length, )
        self.encoder_prompt_attention_mask = torch.ones(self.encoder_prompt_length)
        
        self.encoder = AutoModel.from_pretrained(
                                        config.encoder_model_name_or_path,
                                        from_tf=bool(".ckpt" in config.encoder_model_name_or_path),
                                        config=config)

    def forward(
        self,
        input_ids:torch.Tensor, 
        attention_mask:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, _ = input_ids.shape

        # shape : (batch, length, embedding_dim)
        input_embeddings = self.embeddings(input_ids)

        # shape : (prompt_length, embedding_dim)
        prompt_embeddings = self.encoder_prompt_embeddings(self.encoder_prompt_inputs)
        # shape : (batch, prompt_length, embedding_dim)
        prompt_embeddings = prompt_embeddings.unsqueeze(0).expand(batch_size, self.encoder_prompt_length, -1)

        # shape : (batch, prompt_length)
        prompt_attention_mask = self.encoder_prompt_attention_mask.unsqueeze(0).expand(batch_size, self.encoder_prompt_length)

        input_embeddings = torch.cat([prompt_embeddings, input_embeddings], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)

        # input_embeddings : (batch, length, embedding_dim)
        # attention_mask   : (batch, length)
        return input_embeddings, attention_mask


## for input-dependent processor
class EncoderInputProcessor(BaseInputProcessor):
    def __init__(self, config, embeddings):
        super().__init__(config, embeddings)

        self.encoder_prompt_length = self.config.prompt_length
        assert self.encoder_prompt_length > 0, f'Prompt length must be greater than 0, got {self.encoder_prompt_length}.'
        
        # PLM encoder
        self.encoder = AutoModel.from_pretrained(config.encoder_model_name_or_path)
        self.encoder_embedding_dim = self.encoder.wte.embedding_dim
        self.encoder_generator = torch.nn.Sequential(torch.nn.Linear(self.encoder_embedding_dim, self.embedding_dim // 2),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(self.embedding_dim // 2, self.embedding_dim * self.encoder_prompt_length))
                

    def forward(
        self,
        input_ids:torch.Tensor, 
        attention_mask:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, _ = input_ids.shape

        # shape : (batch, length, embedding_dim)
        input_embeddings = self.embeddings(input_ids)

        ## input dependent prompt ##
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_embeddings = encoder_outputs.last_hidden_state

        sequence_lengths = torch.ne(attention_mask, 0).sum(-1) - 1
        input_dependent_representation = encoder_embeddings[range(batch_size), sequence_lengths]

        # shape : (batch, embedding_dim * encoder_prompt_length)
        input_dependent_representation = self.encoder_generator(input_dependent_representation)
        # shape : (batch, encoder_prompt_length, embedding_dim)
        input_dependent_representation = input_dependent_representation.view(batch_size, self.encoder_prompt_length, self.embedding_dim)
        # shape : (batch, encoder_prompt_length)
        input_dependent_attention_mask = torch.ones((batch_size, self.encoder_prompt_length)).to(input_dependent_representation.device)

        # shape : (batch, prompt+max_length, embedding_dim)
        input_embeddings = torch.cat([input_dependent_representation, input_embeddings], dim=1)
        # shape : (batch, prompt+max_length)
        attention_mask = torch.cat([input_dependent_attention_mask, attention_mask], dim=1)


        # input_embeddings : (batch, length, embedding_dim)
        # attention_mask   : (batch, length)
        return input_embeddings, attention_mask
