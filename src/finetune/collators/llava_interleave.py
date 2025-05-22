import re
from typing import Dict, List, Sequence, Union

import numpy as np
import PIL
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.models.llava.processing_llava import LlavaProcessorKwargs
from transformers.utils import logging

from . import register_collator
from .base import BaseDataCollator

from src.utils import get_vision_chat_message


logger = logging.get_logger(__name__)


# slightly different from https://huggingface.co/llava-hf/llava-interleave-qwen-0.5b-hf/blob/main/chat_template.json
# to include <|im_end|> of assistant's response as labels
template = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% if message['role'] != 'assistant' %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ '\n' + content['text'] }}"
    "{% endfor %}"
    "{% else %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{% generation %}"
    "{{ '\n' + content['text'] }}"
    "{{'<|im_end|>' + '\n'}}"
    "{% endgeneration %}"
    "{% endfor %}"
    "{% endif %}"
    "{% if message['role'] != 'assistant' %}"
    "{{'<|im_end|>' + '\n'}}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


@register_collator("llava-interleave")
class LLaVAInterleaveDataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        
        _input_ids = []
        _assistant_masks = []
        max_len = 0
        for system_prompt, cur_convs in zip(system_prompts, conversations):
            cur_input_ids = []
            cur_text = get_vision_chat_message(system_prompt, cur_convs)
            temp = self.processor.apply_chat_template(
                cur_text,
                chat_template=template,
                add_generation_prompt=False,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False # the assistant tokens mask seems wrong when truncation is enabled
            )
            cur_input_ids = temp["input_ids"]
            cur_assistant_masks = torch.tensor(temp["assistant_masks"], dtype=torch.bool).unsqueeze(0)

            max_len = max(max_len, cur_input_ids.shape[1])
            _input_ids.append(cur_input_ids)
            _assistant_masks.append(cur_assistant_masks)

        input_ids = []
        labels = []
        max_len = min(max_len, self.tokenizer.model_max_length)
        for cur_input_ids, cur_assistant_masks in zip(_input_ids, _assistant_masks):

            # manual truncation
            if cur_input_ids.shape[1] > max_len:
                cur_input_ids = cur_input_ids[:, :max_len]
                cur_assistant_masks = cur_assistant_masks[:, :max_len]
            cur_labels = cur_input_ids.clone()

            if self.mask_question_tokens:
                assert cur_labels.shape == cur_assistant_masks.shape, "Label and mask shapes do not match"
                cur_labels = torch.where(cur_assistant_masks, cur_labels, self.IGNORE_TOKEN_ID)
            assert cur_input_ids.shape == cur_labels.shape, "Input and label shapes do not match"

            # padding
            if cur_input_ids.shape[1] < max_len:
                cur_input_ids = torch.cat([
                    cur_input_ids,
                    torch.full(
                        (cur_input_ids.shape[0], max_len - cur_input_ids.shape[1]),
                        self.PAD_TOKEN_ID,
                        dtype=cur_input_ids.dtype,
                        device=cur_input_ids.device
                    )
                ], dim=1)
                cur_labels = torch.cat([
                    cur_labels,
                    torch.full(
                        (cur_labels.shape[0], max_len - cur_labels.shape[1]),
                        self.IGNORE_TOKEN_ID,
                        dtype=cur_labels.dtype,
                        device=cur_labels.device
                    )
                ], dim=1)

            input_ids.append(cur_input_ids)
            labels.append(cur_labels)

        input_ids = torch.cat(input_ids)
        labels = torch.cat(labels)

        images = [x for instance in instances for x in instance["images"]]
        vision_inputs = self.processor.image_processor(images, return_tensors="pt")

        return dict(
            **vision_inputs,
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.PAD_TOKEN_ID),
        )