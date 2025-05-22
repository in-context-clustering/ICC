from typing import Tuple

from transformers import AutoProcessor, AutoModelForCausalLM, PreTrainedTokenizer, AutoConfig, LlamaForCausalLM

from . import register_loader
from .base import BaseModelLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional


@register_loader("llama-3")
class LLaMA3ModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer, AutoProcessor, AutoConfig]:
        config = AutoConfig.from_pretrained(self.model_local_path)
        if 'Instruct' not in self.model_hf_path:
            processor_path = self.model_hf_path+'-Instruct'
        processor = AutoProcessor.from_pretrained(processor_path, add_eos_token=True)
        tokenizer = processor
        if load_model:
            model = LlamaForCausalLM.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs,
            )
        else:
            model = None
        return model, tokenizer, processor, config


    