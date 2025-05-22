from typing import Dict, List
from collections import OrderedDict

from collators import COLLATORS
from datasets import TO_LOAD_IMAGE
from loaders import LOADERS


MODULE_KEYWORDS: Dict[str, Dict[str, List]] = {
    "llava-interleave": {
        "vision_encoder": ["vision_tower"],
        "vision_projector": ["multi_modal_projector"],
        "llm": ["language_model"]
    },
    "llama-3.1-emb": {
        "vision_encoder": ["vision_model"],
        "vision_projector": ["multi_modal_projector"],
        "llm": ["model"]
    },
    "llama-3": {
        "llm": ["model"]
    }
}


MODEL_HF_PATH = OrderedDict()
MODEL_FAMILIES = OrderedDict()


def register_model(model_id: str, model_family_id: str, model_hf_path: str) -> None:
    if model_id in MODEL_HF_PATH or model_id in MODEL_FAMILIES:
        raise ValueError(f"Duplicate model_id: {model_id}")
    MODEL_HF_PATH[model_id] = model_hf_path
    MODEL_FAMILIES[model_id] = model_family_id



register_model(
    model_id="llava-interleave-qwen-7b",
    model_family_id="llava-interleave",
    model_hf_path="llava-hf/llava-interleave-qwen-7b-hf"
)

register_model(
    model_id="llama-3.1-8b-emb",
    model_family_id="llama-3.1-emb",
    model_hf_path="meta-llama/Llama-3.1-8B"
)

register_model(
    model_id="llama-3.1-8b",
    model_family_id="llama-3",
    model_hf_path="meta-llama/Llama-3.1-8B"
)

register_model(
    model_id="llama-3.2-3b",
    model_family_id="llama-3",
    model_hf_path="meta-llama/Llama-3.2-3B"
)

register_model(
    model_id="llama-3.2-1b",
    model_family_id="llama-3",
    model_hf_path="meta-llama/Llama-3.2-1B"
)

register_model(
    model_id="llama-3.1-8-instruct",
    model_family_id="llama-3",
    model_hf_path="meta-llama/Llama-3.1-8B-Instruct"
)

register_model(
    model_id="llama-3.2-3b-instruct",
    model_family_id="llama-3",
    model_hf_path="meta-llama/Llama-3.2-3B-Instruct"
)

register_model(
    model_id="llama-3.2-1b-instruct",
    model_family_id="llama-3",
    model_hf_path="meta-llama/Llama-3.2-1B-Instruct"
)


# sanity check
for model_family_id in MODEL_FAMILIES.values():
    assert model_family_id in COLLATORS, f"Collator not found for model family: {model_family_id}"
    assert model_family_id in LOADERS, f"Loader not found for model family: {model_family_id}"
    assert model_family_id in MODULE_KEYWORDS, f"Module keywords not found for model family: {model_family_id}"
    assert model_family_id in TO_LOAD_IMAGE, f"Image loading specification not found for model family: {model_family_id}"


if __name__ == "__main__":
    temp = "Model ID"
    ljust = 30
    print("Supported models:")
    print(f"  {temp.ljust(ljust)}: HuggingFace Path")
    print("  ------------------------------------------------")
    for model_id, model_hf_path in MODEL_HF_PATH.items():
        print(f"  {model_id.ljust(ljust)}: {model_hf_path}")
