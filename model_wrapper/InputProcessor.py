

from typing import Tuple, Dict, List

import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer

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


## for Prompt-tuning ##
class PromptInputProcessor(BaseInputProcessor):
    def __init__(self, config, embeddings):
        super().__init__(config, embeddings)

        self.encoder_prompt_length = self.config.prompt_length
        assert self.encoder_prompt_length > 0, f'Prompt length must be greater than 0, got {self.encoder_prompt_length}.'

        self.encoder_prompt_embeddings = torch.nn.Embedding(self.encoder_prompt_length, self.embedding_dim)

                
    def forward(
        self,
        input_ids:torch.Tensor, 
        attention_mask:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, _ = input_ids.shape

        # shape : (batch, length, embedding_dim)
        input_embeddings = self.embeddings(input_ids)

        encoder_prompt_inputs = torch.LongTensor(list(range(self.encoder_prompt_length))).to(input_embeddings.device)
        # shape : (prompt_length, embedding_dim)
        prompt_embeddings = self.encoder_prompt_embeddings(encoder_prompt_inputs)
        # shape : (batch, prompt_length, embedding_dim)
        prompt_embeddings = prompt_embeddings.unsqueeze(0).expand(batch_size, self.encoder_prompt_length, -1)

        # shape : (batch, prompt_length)
        encoder_prompt_attention_mask = torch.ones(self.encoder_prompt_length).to(input_embeddings.device)
        prompt_attention_mask = encoder_prompt_attention_mask.unsqueeze(0).expand(batch_size, self.encoder_prompt_length)

        input_embeddings = torch.cat([prompt_embeddings, input_embeddings], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)

        # input_embeddings : (batch, length, embedding_dim)
        # attention_mask   : (batch, length)
        return input_embeddings, attention_mask


class ReparameterizedInputProcessor(BaseInputProcessor):
    def __init__(self, config, embeddings):
        super().__init__(config, embeddings)

        self.encoder_prompt_length = self.config.prompt_length
        assert self.encoder_prompt_length > 0, f'Prompt length must be greater than 0, got {self.encoder_prompt_length}.'

        # This is used to match the same embedding dim as EncoderInputProcessor
        encoder_config = AutoConfig.from_pretrained(self.config.encoder_model_name_or_path)
        self.encoder_embedding_dim = encoder_config.n_embd
        # only one input embedding 
        # works like the [CLS] embedding in BERT/RoBERTa
        # or last hidden state in GPT2
        self.encoder_prompt_embeddings = torch.nn.Embedding(1, self.encoder_embedding_dim)
        self.encoder_generator = torch.nn.Sequential(torch.nn.Linear(self.encoder_embedding_dim, self.encoder_embedding_dim // 2),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(self.encoder_embedding_dim // 2, self.embedding_dim * self.encoder_prompt_length))
    
    def forward(
        self,
        input_ids:torch.Tensor, 
        attention_mask:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, length = input_ids.shape

        # shape : (batch, length, embedding_dim)
        input_embeddings = self.embeddings(input_ids)

        ## Prompt tokens ##
        # shape : (1, )
        encoder_prompt_inputs = torch.LongTensor(list(range(1))).to(input_embeddings.device)
        # shape : (batch, 1)
        encoder_prompt_inputs = encoder_prompt_inputs.unsqueeze(0).expand(batch_size, 1)
                
        # shape : (batch, 1, encoder_embedding_dim)
        prompt_embeddings = self.encoder_prompt_embeddings(encoder_prompt_inputs)

        # shape : (batch, embedding_dim * encoder_prompt_length)
        final_prompt_embeddings = self.encoder_generator(prompt_embeddings)
        # shape : (batch, encoder_prompt_length, embedding_dim)
        final_prompt_embeddings = final_prompt_embeddings.reshape(batch_size, self.encoder_prompt_length, self.embedding_dim)
        

        # prompt attention mask
        # shape : (batch, encoder_prompt_length)
        final_attention_mask = torch.ones(batch_size, self.encoder_prompt_length).to(final_prompt_embeddings.device)

        # shape : (batch, prompt+max_length, embedding_dim)
        final_prompt_embeddings = torch.cat([final_prompt_embeddings, input_embeddings], dim=1)
        # shape : (batch, prompt+max_length)
        final_attention_mask = torch.cat([final_attention_mask, attention_mask], dim=1)

        assert batch_size == final_prompt_embeddings.shape[0]
        assert length+self.encoder_prompt_length == final_prompt_embeddings.shape[1]
        assert batch_size == final_attention_mask.shape[0]
        assert length+self.encoder_prompt_length == final_attention_mask.shape[1]

        # input_embeddings : (batch, length, embedding_dim)
        # attention_mask   : (batch, length)
        return final_prompt_embeddings, final_attention_mask


## BASELINE ##
## Architecture from IDPG
## for input-dependent processor
class EncoderInputProcessor(BaseInputProcessor):
    def __init__(self, config, embeddings):
        super().__init__(config, embeddings)

        self.encoder_prompt_length = self.config.prompt_length
        assert self.encoder_prompt_length > 0, f'Prompt length must be greater than 0, got {self.encoder_prompt_length}.'
        
        self.encoder_pooling = config.encoder_pooling

        # PLM encoder
        self.encoder = AutoModel.from_pretrained(config.encoder_model_name_or_path)
        self.encoder_embedding_dim = self.encoder.wte.embedding_dim
        self.encoder_generator = torch.nn.Sequential(torch.nn.Linear(self.encoder_embedding_dim, self.encoder_embedding_dim // 2),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(self.encoder_embedding_dim // 2, self.embedding_dim * self.encoder_prompt_length))
                
    def forward(
        self,
        input_ids:torch.Tensor, 
        attention_mask:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, length = input_ids.shape

        ## original input ##
        # shape : (batch, length, embedding_dim)
        input_embeddings = self.embeddings(input_ids)

        ## input dependent prompt ##
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # shape : (batch, length, encoder_embedding_dim)
        encoder_embeddings = encoder_outputs.last_hidden_state

        sequence_lengths = torch.ne(attention_mask, 0).sum(-1) - 1
        if self.encoder_pooling == "last_hidden_state":
            # shape : (batch, encoder_embedding_dim)
            input_dependent_representation = encoder_embeddings[range(batch_size), sequence_lengths]
        elif self.encoder_pooling == "mean":
            input_dependent_representation = []
            for i in range(batch_size):
                last_index = sequence_lengths[i]
                # append shape : (1, encoder_embedding_dim)
                input_dependent_representation.append(torch.mean(encoder_embeddings[i, :last_index], dim=0, keepdim=True))
            # shape : (batch, encoder_embedding_dim)
            input_dependent_representation = torch.cat(input_dependent_representation, dim=0)
        elif self.encoder_pooling == "max":
            input_dependent_representation = []
            for i in range(batch_size):
                last_index = sequence_lengths[i]
                # append shape : (1, encoder_embedding_dim)
                input_dependent_representation.append(torch.max(encoder_embeddings[i, :last_index], dim=0, keepdim=True).values)
            # shape : (batch, encoder_embedding_dim)
            input_dependent_representation = torch.cat(input_dependent_representation, dim=0)




        # shape : (batch, embedding_dim * encoder_prompt_length)
        input_dependent_representation = self.encoder_generator(input_dependent_representation)
        # shape : (batch, encoder_prompt_length, embedding_dim)
        expanded_input_dependent_representation = input_dependent_representation.reshape(batch_size, self.encoder_prompt_length, self.embedding_dim)
                
        # prompt attention mask
        # shape : (batch, encoder_prompt_length)
        input_dependent_attention_mask = torch.ones((batch_size, self.encoder_prompt_length)).to(expanded_input_dependent_representation.device)

        # shape : (batch, prompt+max_length, embedding_dim)
        final_input_embeddings = torch.cat([expanded_input_dependent_representation, input_embeddings], dim=1)
        # shape : (batch, prompt+max_length)
        final_attention_mask = torch.cat([input_dependent_attention_mask, attention_mask], dim=1)

        assert batch_size == final_input_embeddings.shape[0]
        assert length+self.encoder_prompt_length == final_input_embeddings.shape[1]
        assert batch_size == final_attention_mask.shape[0]
        assert length+self.encoder_prompt_length == final_attention_mask.shape[1]

        # input_embeddings : (batch, length, embedding_dim)
        # attention_mask   : (batch, length)
        return final_input_embeddings, final_attention_mask


## We use this encoder to check if using input tokens for generating prompts is meaningful
## If EncoderInputProcessor shows a better performance than PromptEncoderInputProcessor
## means using an input-dependent prompt is helpful ! (our goal)
## for Prompt+Encoder-tuning ##
class PromptEncoderInputProcessor(EncoderInputProcessor):
    def __init__(self, config, embeddings):
        super().__init__(config, embeddings)

        self.encoder_prompt_length = self.config.prompt_length
        assert self.encoder_prompt_length > 0, f'Prompt length must be greater than 0, got {self.encoder_prompt_length}.'
        
        # PLM encoder
        self.encoder = AutoModel.from_pretrained(config.encoder_model_name_or_path)
        self.encoder_embedding_dim = self.encoder.wte.embedding_dim
        self.encoder_generator = torch.nn.Sequential(torch.nn.Linear(self.encoder_embedding_dim, self.encoder_embedding_dim // 2),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(self.encoder_embedding_dim // 2, self.embedding_dim * self.encoder_prompt_length))
        
        # shape : (prompt_length, encoder_embedding_dim)
        self.encoder_prompt_embeddings = torch.nn.Embedding(self.encoder_prompt_length, self.encoder_embedding_dim)

    def forward(
        self,
        input_ids:torch.Tensor, 
        attention_mask:torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, length = input_ids.shape

        # shape : (batch, length, embedding_dim)
        input_embeddings = self.embeddings(input_ids)

        ## Prompt tokens ##
        encoder_prompt_inputs = torch.LongTensor(list(range(self.encoder_prompt_length))).to(input_embeddings.device)
        encoder_prompt_inputs = encoder_prompt_inputs.unsqueeze(0).expand(batch_size, self.encoder_prompt_length)
                
        # shape : (prompt_length, encoder_embedding_dim)
        prompt_embeddings = self.encoder_prompt_embeddings(encoder_prompt_inputs)
        
        # shape : (batch, prompt_length)
        prompt_attention_mask = torch.ones(batch_size, self.encoder_prompt_length).to(input_embeddings.device)

        ## prompt tokens -> PLM encoder ##
        encoder_outputs = self.encoder(inputs_embeds=prompt_embeddings, attention_mask=prompt_attention_mask)
        # shape : (batch, length, encoder_embedding_dim)
        encoder_embeddings = encoder_outputs.last_hidden_state

        sequence_lengths = torch.ne(prompt_attention_mask, 0).sum(-1) - 1
        # shape : (batch, encoder_embedding_dim)
        input_dependent_representation = encoder_embeddings[range(batch_size), sequence_lengths]

        # shape : (batch, embedding_dim * encoder_prompt_length)
        input_dependent_representation = self.encoder_generator(input_dependent_representation)
        # shape : (batch, encoder_prompt_length, embedding_dim)
        expanded_input_dependent_representation = input_dependent_representation.reshape(batch_size, self.encoder_prompt_length, self.embedding_dim)
            
        # prompt attention mask
        # shape : (batch, encoder_prompt_length)
        input_dependent_attention_mask = torch.ones(batch_size, self.encoder_prompt_length).to(expanded_input_dependent_representation.device)

        # shape : (batch, prompt+max_length, embedding_dim)
        final_input_embeddings = torch.cat([expanded_input_dependent_representation, input_embeddings], dim=1)
        # shape : (batch, prompt+max_length)
        final_attention_mask = torch.cat([input_dependent_attention_mask, attention_mask], dim=1)

        assert batch_size == final_input_embeddings.shape[0]
        assert length+self.encoder_prompt_length == final_input_embeddings.shape[1]
        assert batch_size == final_attention_mask.shape[0]
        assert length+self.encoder_prompt_length == final_attention_mask.shape[1]

        # input_embeddings : (batch, length, embedding_dim)
        # attention_mask   : (batch, length)
        return final_input_embeddings, final_attention_mask


## OURS ##
class GenerationProcessor(BaseInputProcessor):
    def __init__(self, config, embeddings, encoder):
        super().__init__(config, embeddings)

        self.encoder_prompt_length = self.config.prompt_length
        assert self.encoder_prompt_length > 0, f'Prompt length must be greater than 0, got {self.encoder_prompt_length}.'
        
        # PLM encoder
        self.encoder = encoder
        self.encoder_embedding = self.encoder.get_input_embeddings()
        self.encoder_embedding_dim = self.encoder_embedding.embedding_dim
        # converts encoder PLM -> inferencing PLM
        self.mapper = torch.nn.Sequential(torch.nn.Linear(self.encoder_embedding_dim, self.encoder_embedding_dim // 2),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(self.encoder_embedding_dim // 2, self.embedding_dim))

        # task dependent prompt
        # TODO : fix?
        #self.encoder_prompt_length = 30
        self.task_dependent_prompt = torch.nn.Embedding(self.encoder_prompt_length, self.encoder_embedding_dim)
        #self.task_dependent_prompt = torch.nn.Embedding(self.encoder_prompt_length, self.encoder_embedding_dim)

    # ## V3 FORWARD ##
    # def forward(
    #     self,
    #     input_ids:torch.Tensor,                 # shape : (batch, max_length)
    #     attention_mask:torch.Tensor,            # shape : (batch, max_length)
    # ) -> Tuple[torch.Tensor, torch.Tensor]:

    #     batch_size, length = input_ids.shape

    #     ## original input ##
    #     # shape : (batch, length, embedding_dim)
    #     input_embeddings = self.embeddings(input_ids)


    #     ## inputs for encoder ##
    #     # shape : (batch, input_length, encoder_embedding_dim)
    #     encoder_input_embeddings = self.encoder_embedding(input_ids)

    #     # list of all the generated prompt embeddings
    #     generated_embeddings = []

    #     kwargs = {
    #             'inputs_embeds' : encoder_input_embeddings,  # shape : (batch, embedding_dim)
    #             'attention_mask' : attention_mask,      # shape : (batch, max_length + 1)
    #             'past_key_values' : None,
    #             'use_cache' : True,
    #         } 

    #     encoder_outputs = self.encoder(**kwargs)
    #     encoder_embeddings = encoder_outputs.last_hidden_state


    #     sequence_lengths = torch.ne(attention_mask, 0).sum(-1) - 1

    #     if self.config.encoder_pooling == "last_hidden_state":
    #         sequence_lengths = torch.ne(attention_mask, 0).sum(-1) - 1
    #         # shape : (batch, encoder_embedding_dim)
    #         input_dependent_representation = encoder_embeddings[range(batch_size), sequence_lengths]
    #     elif self.config.encoder_pooling == "mean":
    #         # shape : (batch, ) -> (batch, 1)
    #         sequence_lengths = torch.ne(attention_mask, 0).sum(-1).unsqueeze(-1)
    #         # shape : (batch, length) -> padding indices
    #         nonzero_indices = torch.ne(attention_mask, 1)
    #         # shape : (batch, length, encoder_embedding_dim)
    #         nonzero_indices = nonzero_indices.unsqueeze(-1).expand(batch_size, length, self.encoder_embedding_dim)

    #         encoder_embeddings[nonzero_indices] = 0
    #         # shape : (batch, encoder_embedding_dim)
    #         encoder_embeddings_sum = torch.sum(encoder_embeddings, dim=1)
    #         # shape : (batch, encoder_embedding_dim)
    #         input_dependent_representation = encoder_embeddings_sum / sequence_lengths
    #     elif self.config.encoder_pooling == "max":
    #         sequence_lengths = torch.ne(attention_mask, 0).sum(-1)
    #         # shape : (batch, length) -> padding indices
    #         nonzero_indices = torch.ne(attention_mask, 1)
    #         # shape : (batch, length, encoder_embedding_dim)
    #         nonzero_indices = nonzero_indices.unsqueeze(-1).expand(batch_size, length, self.encoder_embedding_dim)
    #         # shape : (batch, length, encdoer_embedding_dim)
    #         encoder_embeddings[nonzero_indices] = float('-inf')
    #         # shape : (batch, 1, encoder_embedding_dim)
    #         input_dependent_representation = torch.max(encoder_embeddings, dim=1).values
    #     # shape : (batch, 1, encoder_embedding_dim)
    #     input_dependent_representation = input_dependent_representation.unsqueeze(1)

    #     # encoder representations
    #     ## Prompt tokens ##
    #     # shape : (prompt_length)
    #     encoder_prompt_inputs = torch.LongTensor(list(range(self.encoder_prompt_length))).to(input_embeddings.device)
    #     # shape : (batch, prompt_length)
    #     encoder_prompt_inputs = encoder_prompt_inputs.unsqueeze(0).expand(batch_size, self.encoder_prompt_length)
    #     # shape : (batch, prompt_length, encoder_embedding_dim)
    #     prompt_embeddings = self.task_dependent_prompt(encoder_prompt_inputs)

    #     # shape : (batch, prompt_length+1)
    #     new_attention_mask = torch.ones(batch_size, self.encoder_prompt_length+1).to(input_embeddings.device)

    #     # shape : (batch, prompt_length+1, encoder_embedding_dim)
    #     new_input_embeddings = torch.cat([prompt_embeddings, input_dependent_representation], dim=1)
        


    #     kwargs = {
    #             'inputs_embeds' : new_input_embeddings,                 # shape : (batch, prompt_length + 1, embedding_dim)
    #             'attention_mask' : new_attention_mask,                  # shape : (batch, prompt_length + 1)
    #             'past_key_values' : None,
    #             'use_cache' : True,
    #         } 

    #     _attention_mask = new_attention_mask


    #     # generate prompts
    #     for generation_index in range(self.encoder_prompt_length):
    #         encoder_outputs = self.encoder(**kwargs)
    #         # generated prompt embedding
    #         # shape : (batch, length, encoder_embedding_dim)
    #         encoder_embeddings = encoder_outputs.last_hidden_state
    #         # get the last index (ignore padding)
    #         sequence_lengths = torch.ne(_attention_mask, 0).sum(-1) - 1
    #         # generated last hidden embedding
    #         # shape : (batch, max_length, embedding_dim) -> (batch, embedding_dim) -> (batch, 1, embedding_dim)
    #         generated_embedding = encoder_embeddings[range(batch_size), sequence_lengths].unsqueeze(1)

    #         # append tensor of (batch, embedding_dim) -> (batch, 1, embedding_dim)
    #         generated_embeddings.append(generated_embedding)
    #         # length : # layers
    #         # past key values (for future generation)
    #         past_key_values = encoder_outputs.past_key_values    

    #         # expand attention mask
    #         # shape : (batch, 1)
    #         _attention_mask = torch.ones(batch_size, 1).to(attention_mask.device)

    #         kwargs = {
    #                 'inputs_embeds' : generated_embedding,  # shape : (batch, embedding_dim)
    #                 'attention_mask' : _attention_mask,      # shape : (batch, max_length + 1)
    #                 'past_key_values' : past_key_values,    # List
    #                 'use_cache' : True
    #             }  

    #     # shape : (batch, encoder_prompt_length, encoder_embedding_dim)
    #     generated_embeddings = torch.cat(generated_embeddings, dim=1)

    #     # shape : (batch, encoder_prompt_length, embedding_dim)
    #     generated_embeddings = self.mapper(generated_embeddings)
    #     # shape : (batch, encoder_prompt_length)
    #     generated_attention_mask = torch.ones(batch_size, self.encoder_prompt_length).to(generated_embeddings)

    #     # shape : (batch, encoder_prompt_length + input_length, embedding_dim)
    #     final_input_embeddings = torch.cat([generated_embeddings, input_embeddings], dim=1)
    #     # shape : (batch, encoder_prompt_length + input_length)
    #     final_attention_mask = torch.cat([generated_attention_mask, attention_mask], dim=1)

    #     assert batch_size == final_input_embeddings.shape[0]
    #     assert length+self.encoder_prompt_length == final_input_embeddings.shape[1]
    #     assert batch_size == final_attention_mask.shape[0]
    #     assert length+self.encoder_prompt_length == final_attention_mask.shape[1]

    #     # input_embeddings : (batch, length, embedding_dim)
    #     # attention_mask   : (batch, length)
    #     return final_input_embeddings, final_attention_mask
 
 
 
    ## V2 FORWARD ##
    ################
    def forward(
        self,
        input_ids:torch.Tensor,                 # shape : (batch, max_length)
        attention_mask:torch.Tensor,            # shape : (batch, max_length)
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, length = input_ids.shape

        ## original input ##
        # shape : (batch, length, embedding_dim)
        input_embeddings = self.embeddings(input_ids)


        # encoder representations
        ## Prompt tokens ##
        # shape : (prompt_length)
        encoder_prompt_inputs = torch.LongTensor(list(range(self.encoder_prompt_length))).to(input_embeddings.device)
        # shape : (batch, prompt_length)
        encoder_prompt_inputs = encoder_prompt_inputs.unsqueeze(0).expand(batch_size, self.encoder_prompt_length)
                
        # shape : (prompt_length, encoder_embedding_dim)
        prompt_embeddings = self.task_dependent_prompt(encoder_prompt_inputs)
        
        # shape : (batch, prompt_length)
        prompt_attention_mask = torch.ones(batch_size, self.encoder_prompt_length).to(input_embeddings.device)

        ## inputs for encoder ##
        # shape : (batch, input_length, encoder_embedding_dim)
        encoder_input_embeddings = self.encoder_embedding(input_ids)

        # shape : (batch, input+prompt length, encoder_embedding_dim)
        encoder_input_embeddings = torch.cat([prompt_embeddings, encoder_input_embeddings], dim=1)
        # shape : (batch, input+prompt length)
        encoder_input_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)


        # list of all the generated prompt embeddings
        generated_embeddings = []

        kwargs = {
                'inputs_embeds' : encoder_input_embeddings,  # shape : (batch, embedding_dim)
                'attention_mask' : encoder_input_attention_mask,      # shape : (batch, max_length + 1)
                'past_key_values' : None,
                'use_cache' : True,
            } 

        _attention_mask = encoder_input_attention_mask
        

        # generate prompts
        for generation_index in range(self.encoder_prompt_length):
            encoder_outputs = self.encoder(**kwargs)
            # generated prompt embedding
            # shape : (batch, length, encoder_embedding_dim)
            encoder_embeddings = encoder_outputs.last_hidden_state
            # get the last index (ignore padding)
            sequence_lengths = torch.ne(_attention_mask, 0).sum(-1) - 1
            # generated last hidden embedding
            # shape : (batch, max_length, embedding_dim) -> (batch, embedding_dim) -> (batch, 1, embedding_dim)
            generated_embedding = encoder_embeddings[range(batch_size), sequence_lengths].unsqueeze(1)

            # append tensor of (batch, embedding_dim) -> (batch, 1, embedding_dim)
            generated_embeddings.append(generated_embedding)
            # length : # layers
            # past key values (for future generation)
            past_key_values = encoder_outputs.past_key_values    

            # expand attention mask
            # shape : (batch, 1)
            _attention_mask = torch.ones(batch_size, 1).to(attention_mask.device)

            kwargs = {
                    'inputs_embeds' : generated_embedding,  # shape : (batch, embedding_dim)
                    'attention_mask' : _attention_mask,      # shape : (batch, max_length + 1)
                    'past_key_values' : past_key_values,    # List
                    'use_cache' : True
                }  

        # shape : (batch, encoder_prompt_length, encoder_embedding_dim)
        generated_embeddings = torch.cat(generated_embeddings, dim=1)

        # shape : (batch, encoder_prompt_length, embedding_dim)
        generated_embeddings = self.mapper(generated_embeddings)
        # shape : (batch, encoder_prompt_length)
        generated_attention_mask = torch.ones(batch_size, self.encoder_prompt_length).to(generated_embeddings)

        # shape : (batch, encoder_prompt_length + input_length, embedding_dim)
        final_input_embeddings = torch.cat([generated_embeddings, input_embeddings], dim=1)
        # shape : (batch, encoder_prompt_length + input_length)
        final_attention_mask = torch.cat([generated_attention_mask, attention_mask], dim=1)

        assert batch_size == final_input_embeddings.shape[0]
        assert length+self.encoder_prompt_length == final_input_embeddings.shape[1]
        assert batch_size == final_attention_mask.shape[0]
        assert length+self.encoder_prompt_length == final_attention_mask.shape[1]

        # input_embeddings : (batch, length, embedding_dim)
        # attention_mask   : (batch, length)
        return final_input_embeddings, final_attention_mask
    ## V2 FORWARD END ##




## V1 FORWARD ##
################
# class GenerationProcessor(BaseInputProcessor):
#     def __init__(self, config, embeddings):
#         super().__init__(config, embeddings)

#         self.encoder_prompt_length = self.config.prompt_length
#         assert self.encoder_prompt_length > 0, f'Prompt length must be greater than 0, got {self.encoder_prompt_length}.'
        
#         # PLM encoder
#         self.encoder = AutoModel.from_pretrained(config.encoder_model_name_or_path)
#         self.encoder_tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_name_or_path)
#         self.encoder_embedding_dim = self.encoder.wte.embedding_dim
#         # converts encoder PLM -> inferencing PLM
#         self.mapper = torch.nn.Sequential(torch.nn.Linear(self.encoder_embedding_dim, self.encoder_embedding_dim // 2),
#                                             torch.nn.ReLU(),
#                                             torch.nn.Linear(self.encoder_embedding_dim // 2, self.embedding_dim))

#         # task dependent prompt
#         self.task_dependent_prompt = torch.nn.Embedding(self.encoder_prompt_length, self.encoder_embedding_dim)

#     def forward(
#         self,
#         input_ids:torch.Tensor,                 # shape : (batch, max_length)
#         attention_mask:torch.Tensor,            # shape : (batch, max_length)
#     ) -> Tuple[torch.Tensor, torch.Tensor]:

#         batch_size, length = input_ids.shape

#         ## original input ##
#         # shape : (batch, length, embedding_dim)
#         input_embeddings = self.embeddings(input_ids)


#         # list of all the generated prompt embeddings
#         generated_embeddings = []

#         kwargs = {
#                 'input_ids' : input_ids,  # shape : (batch, embedding_dim)
#                 'attention_mask' : attention_mask,      # shape : (batch, max_length + 1)
#                 'past_key_values' : None,
#                 'use_cache' : True,
#             } 

#         _attention_mask = attention_mask

#         # generate prompts
#         for generation_index in range(self.encoder_prompt_length):
#             encoder_outputs = self.encoder(**kwargs)
#             # generated prompt embedding
#             # shape : (batch, length, encoder_embedding_dim)
#             encoder_embeddings = encoder_outputs.last_hidden_state
#             # get the last index (ignore padding)
#             sequence_lengths = torch.ne(_attention_mask, 0).sum(-1) - 1
#             # generated last hidden embedding
#             # shape : (batch, max_length, embedding_dim) -> (batch, embedding_dim) -> (batch, 1, embedding_dim)
#             generated_embedding = encoder_embeddings[range(batch_size), sequence_lengths].unsqueeze(1)

#             # append tensor of (batch, embedding_dim) -> (batch, 1, embedding_dim)
#             generated_embeddings.append(generated_embedding)
#             # length : # layers
#             # past key values (for future generation)
#             past_key_values = encoder_outputs.past_key_values    

#             # expand attention mask
#             # shape : (batch, 1)
#             _attention_mask = torch.ones(batch_size, 1).to(attention_mask.device)

#             kwargs = {
#                     'inputs_embeds' : generated_embedding,  # shape : (batch, embedding_dim)
#                     'attention_mask' : _attention_mask,      # shape : (batch, max_length + 1)
#                     'past_key_values' : past_key_values,    # List
#                     'use_cache' : True
#                 }  

#         # shape : (batch, encoder_prompt_length, encoder_embedding_dim)
#         generated_embeddings = torch.cat(generated_embeddings, dim=1)

#         # shape : (batch, encoder_prompt_length, embedding_dim)
#         generated_embeddings = self.mapper(generated_embeddings)
#         # shape : (batch, encoder_prompt_length)
#         generated_attention_mask = torch.ones(batch_size, self.encoder_prompt_length).to(generated_embeddings)

#         # shape : (batch, encoder_prompt_length + input_length, embedding_dim)
#         final_input_embeddings = torch.cat([generated_embeddings, input_embeddings], dim=1)
#         # shape : (batch, encoder_prompt_length + input_length)
#         final_attention_mask = torch.cat([generated_attention_mask, attention_mask], dim=1)

#         assert batch_size == final_input_embeddings.shape[0]
#         assert length+self.encoder_prompt_length == final_input_embeddings.shape[1]
#         assert batch_size == final_attention_mask.shape[0]
#         assert length+self.encoder_prompt_length == final_attention_mask.shape[1]

#         # input_embeddings : (batch, length, embedding_dim)
#         # attention_mask   : (batch, length)
#         return final_input_embeddings, final_attention_mask

## V1 FORWARD END ##

