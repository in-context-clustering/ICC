import torch
import os
import numpy as np
from tqdm import tqdm
import traceback
import argparse
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import KMeans
import certifi
import openai
from openai import OpenAI

def hf_generate(model, system_prompt, user_prompt, tokenizer, maxlength):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").cuda()
    with torch.inference_mode():
        outputs = model.generate(inputs, temperature=0.01, max_new_tokens=maxlength)
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

def openai_generate(client, system_prompt, user_prompt, model_name, maxlength):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    params = {
            "messages": messages,
            "model": model_name
        }
    if 'o3' not in model_name:
        params["max_tokens"] =  maxlength
        params["temperature"] = 0
    else:
        params["max_completion_tokens"] =  maxlength
        # 'temperature' does not support 0 with this model. Only the default (1) value is supported.

    num_retries = 0
    while True:
        try:
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        except openai.RateLimitError:
            num_retries += 1
            if num_retries > 5:
                raise Exception(
                    f"Got openai.RateLimitError. Maximum number of retries exceeded."
                )
            time.sleep(1)

def kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    y_pred = kmeans.fit_predict(X)
    return y_pred.tolist()

def clustering_factory(modelname, num_clusters, maxlength):
    if modelname == 'kmeans':
        return lambda x: kmeans(np.array(x['X']), num_clusters)
    if '/scratch' in modelname:
        model = AutoModelForCausalLM.from_pretrained(modelname, token=access_token).cuda()
        tokenizer = AutoTokenizer.from_pretrained(modelname)
        return lambda x: hf_generate(model, x['system_prompt'], x['conversations'][0]['value'], tokenizer, maxlength)
    if 'llama' in modelname:
        model = AutoModelForCausalLM.from_pretrained(modelname).cuda()
        if "Instruct" not in modelname:
            tokenizer = AutoTokenizer.from_pretrained(f"{modelname}-Instruct")
        else:
            tokenizer = AutoTokenizer.from_pretrained(modelname)
        return lambda x: hf_generate(model, x['system_prompt'], x['conversations'][0]['value'], tokenizer, maxlength)
    if 'Qwen' in modelname:
        model = AutoModelForCausalLM.from_pretrained(modelname).cuda()
        tokenizer = AutoTokenizer.from_pretrained(modelname)
        return lambda x: hf_generate(model, x['system_prompt'], x['conversations'][0]['value'], tokenizer, maxlength)
    if 'gpt' in modelname or 'o3' in modelname:
        client = OpenAI()
        return lambda x: openai_generate(client, x['system_prompt'], x['conversations'][0]['value'], modelname, maxlength)
    if 'deepseek' in modelname:
        client = OpenAI(base_url="https://api.deepseek.com")
        return lambda x: openai_generate(client, x['system_prompt'], x['conversations'][0]['value'], modelname, maxlength)


def main(args):
    print(f'Evaluating {args.modelname}...', flush=True)
    name = args.modelname.split('/')[-1] if '/' in args.modelname else args.modelname
    if '/scratch' in args.modelname:
        name = f"finetuned_{args.modelname.split('/')[-2]}"
    directory = f"{args.output_directory}/{name}"
    prefix = args.data_path.split('/')[-1].split('.')[0]

    if not os.path.exists(directory):
        os.makedirs(directory)
    elif os.path.exists(f"{directory}/{prefix}_{name}.json"):
        print(f'{directory}/{prefix}_{name}.json Already done!', flush=True)
        return

    with open(args.data_path, 'r') as f:
        print(f'Evaluating on {args.data_path}', flush=True)
        data = json.load(f)

    pred_func = clustering_factory(args.modelname, args.c, args.maxlength*5)
    output = []
    for row in tqdm(data):
        pred = pred_func(row)
        output.append(pred)
    
    with open(f"{directory}/{prefix}_{name}.json", "w") as outfile: 
        json.dump(output, outfile)

    print('#'*20, flush=True)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate numeric ICC')
    parser.add_argument('-c', required=True, type=int)
    parser.add_argument('--maxlength', default=50, type=int)
    parser.add_argument('--modelname', required=True, type=str)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--output_directory', required=True, type=str)
    args = parser.parse_args()
    main(args)