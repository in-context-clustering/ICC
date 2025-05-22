import json
import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import argparse
import traceback
from sklearn.cluster import KMeans, AgglomerativeClustering
from transformers import AutoProcessor, LlavaForConditionalGeneration, PreTrainedTokenizer, AutoConfig
from src.utils import compute_clustering_accuracy, postprocess, extract_cluster_info, move_to_cuda, get_vision_chat_message
from src.finetune.loaders.llava_interleave import LLaVAWrapper

def get_image_features(model, inputs):
    with torch.inference_mode():
        with torch.amp.autocast('cuda'):
            features = model.vision_tower(inputs)['last_hidden_state'] # [length, 729, 1152]
    return features.mean(axis=1).cpu().numpy() # [length, 1152]

def kmeans(model, inputs, num_clusters):
    image_emb = get_image_features(model, inputs)
    method = KMeans(n_clusters=num_clusters)
    preds = method.fit_predict(image_emb)
    return preds.tolist()

def agglomerative(model, inputs, num_clusters):
    image_emb = get_image_features(model, inputs)
    method = AgglomerativeClustering(n_clusters=num_clusters)
    y_kmeans = method.fit_predict(image_emb)
    return preds.tolist()

def llm_generate(model, processor, inputs, maxlength):
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            output = model.generate(**inputs, 
                                    max_new_tokens=maxlength, 
                                    do_sample=False,
                                    temperature=0
                                   )
    response =  processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response

def load_images_from_hpc(filenames, directory=""):
    images = []
    if '/scratch' not in filenames[0]:
        directory="/imagenet21k/"
    for filename in filenames:
        img = Image.open(f'{directory}{filename}').convert("RGB")
        images.append(img)
    return images

def process(processor, ex, imageonly=False):
    images = load_images_from_hpc(ex['image'])
    processed_images = processor.image_processor(images, return_tensors="pt")['pixel_values']
    if imageonly: 
        return processed_images.cuda()
    else:
        messages = get_vision_chat_message(ex["system_prompt"], ex["conversations"][:-1])
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", tokenize=True, return_dict=True, truncation=False)
        inputs['pixel_values'] = processed_images
        return  move_to_cuda(inputs)

def clustering_factory(modelname, num_clusters, maxlength, basemodel = "llava-hf/llava-interleave-qwen-7b-hf"):
    if modelname == 'kmeans' or modelname == 'agg':
        model = LlavaForConditionalGeneration.from_pretrained(
            basemodel, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
        ).to('cuda')
        processor = AutoProcessor.from_pretrained(basemodel)
    else:
        patch_size = int(modelname.split('patch')[-1].split('_')[0])
        model = LLaVAWrapper.from_pretrained(
            modelname, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            patch_size = patch_size
        ).to('cuda')
        processor = AutoProcessor.from_pretrained(basemodel)
        processor.patch_size = patch_size
    if modelname == 'kmeans':
        return lambda x: kmeans(model, process(processor, x, imageonly=True), num_clusters)
    if modelname == 'agg':
        return lambda x: agglomerative(model, process(processor, x, imageonly=True), num_clusters)
    if '/scratch' in modelname:
        return lambda x: llm_generate(model, processor, process(processor, x), maxlength)


def main(args):
    if 'cond' in args.data_path:
        task = "image_cond"
    else:
        task = "image"
    
    prefix = args.data_path.split('/')[-1].split('.')[0]
    if '/scratch' in args.modelname:
        name = f"finetuned_{args.modelname.split('/')[-2]}"
    else:
        name = args.modelname

    output_path = f"{args.output_dir}/{task}/{name}/{prefix}_{name}.json"
    num_clusters = args.c
    
    print(output_path, flush=True)
    if not os.path.exists(f"{args.output_dir}/{task}/{name}"):
        os.makedirs(f"{args.output_dir}/{task}/{name}")
    elif os.path.exists(output_path):
        print('Already done!')
        return

    with open(args.data_path, 'r') as f:
        print(f'Data {num_clusters}c: {args.data_path} \n {"#"*20}', flush=True)
        data = json.load(f)

    
    pred_func = clustering_factory(args.modelname, num_clusters, 256)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-interleave-qwen-7b-hf")
    output = []
    for ex in tqdm(data):
        pred = pred_func(ex)
        output.append(pred)

    with open(output_path, "w") as outfile: 
        json.dump(output, outfile)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Eval images')
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--modelname', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('-c', required=True, type=int)
    args = parser.parse_args()
    main(args)