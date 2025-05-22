from typing import Tuple

from transformers import AutoProcessor, AutoModelForCausalLM, PreTrainedTokenizer, AutoConfig, LlamaForCausalLM

from . import register_loader
from .base import BaseModelLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional


@register_loader("llama-3.1-emb")
class LLaMA31EmbModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer, AutoProcessor, AutoConfig]:
        config = AutoConfig.from_pretrained(self.model_local_path)
        processor = AutoProcessor.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', add_eos_token=True)
        tokenizer = processor
        tokenizer.add_special_tokens({'additional_special_tokens': ['<num>']})
        if load_model:
            model = LLaMAWrapper.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs,
            )
            model.num_token_index = tokenizer.convert_tokens_to_ids('<num>')
            model.resize_token_embeddings(len(tokenizer))
        else:
            model = None
        return model, tokenizer, processor, config

class LLaMAWrapper(LlamaForCausalLM):

    def __init__(self, config, **kwargs):
        super(LLaMAWrapper, self).__init__(config)
        self.config = config
        self.num_token_index = 128256
        self.num_proj = nn.Linear(4096, 4096)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds = None,
        numeric_data = None,
        **kwargs
    ):
        if numeric_data is None:
            return super().forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                **kwargs
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            special_tok_mask = (input_ids == self.num_token_index).unsqueeze(-1)
            special_tok_mask = special_tok_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            numeric_data = numeric_data.to(inputs_embeds.device, inputs_embeds.dtype)
            numeric_data = self.num_proj(numeric_data).to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_tok_mask, numeric_data)
            

        return super().forward(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                 **kwargs
            )


    def generate(self, input_ids, inputs_embeds = None, numeric_data=None, **kwargs):

        inputs_embeds = self.get_input_embeddings()(input_ids)
        special_tok_mask = (input_ids == self.num_token_index).unsqueeze(-1)
        special_tok_mask = special_tok_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        numeric_data = numeric_data.to(inputs_embeds.device, inputs_embeds.dtype)
        numeric_data = self.num_proj(numeric_data).to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_tok_mask, numeric_data)
        return super().generate(input_ids=None, inputs_embeds=inputs_embeds,  **kwargs)


    