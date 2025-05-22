from typing import Tuple

from transformers import AutoProcessor, LlavaForConditionalGeneration, PreTrainedTokenizer, AutoConfig

from . import register_loader
from .base import BaseModelLoader
import torch

@register_loader("llava-interleave")
class LLaVAInterleaveModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[LlavaForConditionalGeneration, PreTrainedTokenizer, AutoProcessor, AutoConfig]:
        if load_model:
            model = LLaVAWrapper.from_pretrained(
                self.model_local_path, 
                self.patch_size,
                **self.loading_kwargs,
            )
            model.config.hidden_size = model.language_model.config.hidden_size # useful for deepspeed
        else:
            model = None

        processor = AutoProcessor.from_pretrained(self.model_hf_path)
        processor.patch_size = self.patch_size
        tokenizer = processor.tokenizer
        config = AutoConfig.from_pretrained(self.model_hf_path)
        return model, tokenizer, processor, config


class LLaVAWrapper(LlavaForConditionalGeneration):

    def __init__(self, config, patch_size, **kwargs):
        super(LLaVAWrapper, self).__init__(config)
        self.patch_size = patch_size
        self.pool = torch.nn.AvgPool2d(patch_size//14)

    def get_image_features(self, pixel_values, **kwargs):
        # can be changed to other feature exractors
        # currently, it's assumed to use the vision encoder of SigLip which has a patch size of 14 (729 tokens per image)
        image_features = self.vision_tower(pixel_values)['last_hidden_state'] #[40, 729, 1152]
        if self.patch_size > 14:
            B, T, D = image_features.shape
            image_features = image_features.permute(0,2,1).reshape(B, D, 27, 27)
            image_features = self.pool(image_features) # N,C,W,H
            image_features = image_features.reshape(B,D,-1).permute(0,2,1)
        image_features = self.multi_modal_projector(image_features).to(torch.bfloat16)
        return image_features