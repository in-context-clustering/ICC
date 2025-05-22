COLLATORS = {}

def register_collator(name):
    def register_collator_cls(cls):
        if name in COLLATORS:
            return COLLATORS[name]
        COLLATORS[name] = cls
        return cls
    return register_collator_cls


from .llava_interleave import LLaVAInterleaveDataCollator
from .llama_3_1_emb import LLaMA31EmbDataCollator
from .llama_3 import LLaMA3DataCollator
