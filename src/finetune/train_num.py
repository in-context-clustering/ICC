import os
from dataclasses import asdict
from pathlib import Path
import yaml
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import transformers
from transformers import Trainer#, deepspeed
from transformers.integrations import deepspeed, WandbCallback
import traceback
import wandb
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional

from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from collators import COLLATORS
from datasets import LazySupervisedDataset
from loaders import LOADERS
from supported_models import MODULE_KEYWORDS
import torch.distributed as dist
from lmms_utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, TrainerWithCustomSampler
)
import warnings
warnings.filterwarnings("ignore")
from src.utils import get_chat_message, compute_clustering_accuracy, postprocess, extract_cluster_info, move_to_cuda
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()


class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, processor, output_dir, max_new_tokens=10000, log_model=None):
        super().__init__()
        self.preprocessed_dataset = []
        self.trainer = trainer
        self.model = trainer.model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.output_dir = output_dir
        self.records_table = wandb.Table(columns=["step", "prompt", "gt", "generation", "acc"] )
        for ex in test_dataset:
            messages = get_chat_message(ex["system_prompt"], ex["conversations"][:1])
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", tokenize=True, return_dict=True, truncation=False)
            info = {}
            info["prompt"] = ex["conversations"][0]
            info["labels"] = ex["conversations"][1]
            self.preprocessed_dataset.append({"inputs": inputs, "info": info})

    def generate(self, inputs):
        #print({v.shape for k,v in inputs.items()}) #{torch.Size([10, 3, 3, 384, 384]), torch.Size([1, 54]), torch.Size([10, 2])}
        inputs = move_to_cuda(inputs)
        with torch.inference_mode():
            with torch.amp.autocast('cuda'):
                output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        response =  self.processor.decode(output[0], skip_special_tokens=True)
        return response.strip()

    def samples_table(self, step):
        avg_acc = []
        for ex in tqdm(self.preprocessed_dataset):
            generation = self.generate(ex["inputs"])
            try:
                labels = eval(ex["info"]["labels"])
                results = postprocess(generation, len(labels))
                acc = compute_clustering_accuracy(labels, results)
            except Exception as e:
                print(traceback.format_exc(), flush=True)
                acc = 0
            self.records_table.add_data(step, ex["info"]["prompt"], ex["info"]["labels"], generation, acc)
            avg_acc.append(acc)
        return {f"eval/acc": sum(avg_acc)/len(avg_acc)}
        
    def on_evaluate(self, args, state, control,  **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        if dist.get_rank() == 0:
            avg_acc = self.samples_table(state.global_step)
            new_table = wandb.Table(
                columns=self.records_table.columns, data=self.records_table.data
            )
            self._wandb.log({"sample_predictions":new_table, **avg_acc})
        safe_save_model_for_hf_trainer(trainer=self.trainer, output_dir=self.output_dir)

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # dumping arguments
    output_dir = getattr(training_args, 'output_dir', None)
    assert output_dir is not None, "output_dir is required"
    args_dir = Path(output_dir) / "arguments"
    args_dir.mkdir(parents=True, exist_ok=True)
    yaml.dump(asdict(model_args), open(args_dir / "model.yaml", "w"))
    yaml.dump(asdict(data_args), open(args_dir / "data.yaml", "w"))
    yaml.dump(asdict(training_args), open(args_dir / "training.yaml", "w"))
    yaml.dump(asdict(lora_args), open(args_dir / "lora.yaml", "w"))

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if int(os.environ.get("WORLD_SIZE", 1)) != 1 else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            raise ValueError("FSDP or ZeRO3 are not incompatible with QLoRA.")

    # llm quantization config (for q-lora)
    bnb_config = None
    if lora_args.use_lora and lora_args.q_lora:
        from transformers import BitsAndBytesConfig
        rank0_print("Quantization for LLM enabled...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4", 
        )
    
    # load model, tokenizer, processor
    rank0_print("Loading model, tokenizer, processor...")
    loader = LOADERS[model_args.model_family_id](
        model_hf_path=model_args.model_hf_path,
        model_local_path=model_args.model_local_path,
        compute_dtype=compute_dtype,
        bnb_config=bnb_config,
        use_flash_attn=training_args.use_flash_attn,
        device_map=device_map,
    )
    model, tokenizer, processor, config = loader.load()
    tokenizer.model_max_length = training_args.model_max_length

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # lora preparation
    llm_keys = MODULE_KEYWORDS[model_args.model_family_id]["llm"]
    if not (lora_args.use_lora or (training_args.train_vision_encoder and lora_args.use_vision_lora)):
        rank0_print("No LoRA enabled...")        
    else:
        named_modules = {n: m for n, m in model.named_modules()}
        lora_modules = []
        full_modules = []
        
        if lora_args.use_lora:
            rank0_print("LoRA for LLM enabled...")
            lora_modules.extend(find_all_linear_names(named_modules, llm_keys))
        else:
            rank0_print("LLM will be fully trained...")
            full_modules.extend(llm_keys)
        
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_modules,
            modules_to_save=full_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
            
        model = get_peft_model(model, lora_config)
        
    # print trainable parameters for inspection
    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(f"\t{name}")

    # load data
    rank0_print("Loading data...")
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        video_folder=data_args.video_folder,
        num_frames=data_args.num_frames,
        model_family_id=model_args.model_family_id,
        user_key=data_args.user_key,
        assistant_key=data_args.assistant_key
    )
    if data_args.eval_data_path:
        rank0_print("Loading eval data...")
        eval_dataset = LazySupervisedDataset(
            data_path=data_args.eval_data_path,
            image_folder=data_args.image_folder,
            video_folder=data_args.video_folder,
            num_frames=data_args.num_frames,
            model_family_id=model_args.model_family_id,
            user_key=data_args.user_key,
            assistant_key=data_args.assistant_key
        )
    else:
        eval_dataset = None
        training_args.eval_strategy = "no"

    # data collator
    data_collator = COLLATORS[model_args.model_family_id](
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        mask_question_tokens=training_args.mask_question_tokens
    )

    training_args.load_best_model_at_end = False

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    wandb_callback = LLMSampleCB(trainer, eval_dataset, processor, output_dir, max_new_tokens=256)
    trainer.add_callback(wandb_callback)
    trainer.train()
    

if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = None # TODO: replace
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_DATA_DIR"] = None # TODO: replace
    os.environ["HUGGINGFACE_TOKEN"] = None # TODO: replace
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    train()