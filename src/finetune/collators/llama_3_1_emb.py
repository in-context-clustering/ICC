import re
from typing import Dict, List, Sequence, Union

import numpy as np
import PIL
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.models.mllama.processing_mllama import MllamaProcessorKwargs
from transformers.utils import logging

from . import register_collator
from .base import BaseDataCollator
from .chat_template_monkey_patch import apply_chat_template


logger = logging.get_logger(__name__)

template = """

{% if messages[0]['role'] == 'system' %}
    {% set offset = 1 %}
{% else %}
    {% set offset = 0 %}
{% endif %}

{{ bos_token }}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'}}
    {% if message['role'] == 'assistant' %}
        {% generation %} {{ message['content'] | trim + '<|eot_id|>' }} {% endgeneration %} 
    {% else %}
        {{ message['content'] | trim }}
    {% endif %}


{% endfor %}

{% if add_generation_prompt %}
    {{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n' }}
{% endif %}

"""




@register_collator("llama-3.1-emb")
class LLaMA31EmbDataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # monkey patch to include bos tokens
        self.tokenizer.apply_chat_template = apply_chat_template.__get__(self.tokenizer)
        
        numeric_inputs = np.array([np.pad(np.array(x), (0, 4096-len(x)), mode='constant', constant_values=0) for instance in instances for x in instance["numbers"]])
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        
        _input_ids = []
        _assistant_masks = []
        max_input_len = 0
        for system_prompt, cur_convs in zip(system_prompts, conversations):
            cur_input_ids = []
            cur_labels = []
            cur_text = []
            if system_prompt is not None:
                cur_text.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            for i, text in enumerate(cur_convs):
                if i % 2 == 0:
                    cur_text.append({
                        "role": "user",
                        "content": text
                    })
                else:
                    cur_text.append({
                        "role": "assistant",
                        "content": text
                    })
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

            # a dirty hack to include eos token as part of the labels
            cur_assistant_masks[0, -1] = True
            
            max_input_len = max(max_input_len, cur_input_ids.shape[1])
            _input_ids.append(cur_input_ids)
            _assistant_masks.append(cur_assistant_masks)

        input_ids = []
        labels = []
        max_len = max_input_len
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

        return dict(
            numeric_data=torch.tensor(numeric_inputs),
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.PAD_TOKEN_ID),
        )