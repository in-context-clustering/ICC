# In-Context Clustering with Large Language Models

## Setup
```
conda create -n icc python=3.10 -y
conda activate icc
cd ICC
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:/path/to/ICC"
```
## Expected Data Format

A json file containing a list of clustering episodes
```
{"image": <a list of image file paths>, #optional
 "system_prompt": <sys_prompt>,
 "conversations":[
    {"from": "human", "value": <instruction + data>},
    {"from": "gpt", "value": <a list of ground-truth cluster labels>}
  ]
}
```

## Quick Start

To reproduce attention visualization:
- check attention_viz.ipynb

To perform spectral clustering using attention:
```
python src/spectral_clustering.py
  -c <number of clusters>
  --data_path <path to the test data file>
  --output_dir <path to the output directory>
  --modelname <model name, such as k-means, HF model ids, or checkpoint path>
```

To evaluate the model:
```
python src/eval/evaluate_image.py
  -c <number of clusters>
  --data_path <path to the test data file>
  --output_dir <path to the output directory>
  --modelname <model name, such as kmeans, agg, HF model ids, or checkpoint path>

python src/eval/evaluate_numeric.py
  -c <number of clusters>
  --data_path <path to the test data file>
  --output_directory <path to the output directory>
  --modelname <model name, such as kmeans, openai models, HF model ids, or checkpoint path>
```

To finetune the model on clustering data: 
- Check src/finetune/scripts and update config with your data paths and compute resources
```
bash src/finetune/scripts/llama_ft_num.sh <learning_rate>
bash src/finetune/scripts/llava_ft_img.sh <learning_rate> <patch_size> <batch_size>
```
