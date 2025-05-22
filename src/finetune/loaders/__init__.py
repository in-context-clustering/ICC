LOADERS = {}

def register_loader(name):
    def register_loader_cls(cls):
        if name in LOADERS:
            return LOADERS[name]
        LOADERS[name] = cls
        return cls
    return register_loader_cls


from .llava_interleave import LLaVAInterleaveModelLoader
from .llama_3_1_emb import LLaMA31EmbModelLoader
from .llama_3 import LLaMA3ModelLoader