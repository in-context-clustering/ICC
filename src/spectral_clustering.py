import transformers
import torch
import os
from transformers import TrainingArguments, AutoTokenizer
import random
import numpy as np
import argparse
from src.utils import extract_list, compute_clustering_accuracy, postprocess, get_chat_message
import traceback
import warnings
import json
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
warnings.filterwarnings('ignore')
    
def hf_forward(inputs, model):
    with torch.inference_mode():
        outputs = model(inputs, output_hidden_states=True)
    return outputs

def get_tokens(tokenizer, x):
    text_list = [x, f' {x}', f'{x} ', f'{x}\n', f' {x}\n']
    return [tokenizer.encode(text)[1] for text in text_list]
    
def get_segments(tokenizer, inputs,points):
    segments = []
    current_segment = []
    for i,token in enumerate(inputs.squeeze().tolist()):
        if token in (get_tokens(tokenizer, ']]') + get_tokens(tokenizer, ']\\')):
            segments.append((min(current_segment), max(current_segment)+1))
            break
        if token in (get_tokens(tokenizer,'[') + get_tokens(tokenizer, '[[')):
            if current_segment:
                segments.append((min(current_segment), max(current_segment)))
            current_segment = [i]
        else:
            current_segment.append(i)
    segments = segments[1:]
    assert len(segments) == len(points)
    return segments

def main(args):
    with open(args.data_path) as f:
        data = json.load(f)
    prefix = args.data_path.split('/')[-1].replace('.json', '')
    model = AutoModelForCausalLM.from_pretrained(args.modelname, output_attentions=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.modelname)
    name = args.modelname.split('/')[-1] if '/' in args.modelname else args.modelname
    if '/scratch' in args.modelname:
        name = f"finetuned_{args.modelname.split('/')[-2]}"
    att = []
    for idx in tqdm(range(len(data))):
        messages = get_chat_message(data[idx]['system_prompt'], data[idx]['conversations'])[:-1]
        inputs = tokenizer.apply_chat_template(messages, temperature=0, add_generation_prompt=True, return_tensors="pt").cuda()
        points = data[idx]['X']
        segments = get_segments(tokenizer, inputs, points)
        outputs = hf_forward(inputs, model)
        acc_list = []
        for l, layer in enumerate(range(32)):
            avg_attention = outputs['attentions'][layer][0].mean(axis=0)
            W = torch.zeros((len(points), len(points)))
            for i in range(len(points)):
                for j in range(i + 1, len(points)): 
                    s1,e1 = segments[i]
                    s2,e2 = segments[j]
                    weight = avg_attention[s2:e2+1, s1:e1+1].mean().item()
                    W[j, i] = weight

            W /= (W.sum(axis=1, keepdim=True)+1e-7)
            for i in range(len(points)):
                for j in range(i + 1, len(points)): 
                    W[j, i] = W[j, i]*j
                    W[i, j] = W[j, i]
            
            clusters = SpectralClustering(n_clusters=args.c, 
                affinity='precomputed', random_state=0).fit_predict(W)
            acc = compute_clustering_accuracy(clusters, data[idx]['gt'])
            acc_list.append(acc)
        att.append(acc_list)

    with open(f"{output_dir}/{prefix}_{name}_sc_results.json", "w") as outfile: 
        json.dump(att, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Data generator')
    parser.add_argument('-c', default=2, type=int)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--modelname', required=True, type=str)
    args = parser.parse_args()
    main(args)