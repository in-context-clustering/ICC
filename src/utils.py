import pandas as pd
import numpy as np
import re
import os
import random
from PIL import Image
import json
import numbers
from collections.abc import Iterable
from sklearn.cluster import KMeans
from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as util_shuffle
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment
import torch

def compute_clustering_accuracy(y_true, y_pred):
    # Hungarian algorithm to find the best label correspondence
    cost_matrix = np.zeros((len(np.unique(y_true)), len(np.unique(y_pred))))
    for i, true_label in enumerate(np.unique(y_true)):
        for j, pred_label in enumerate(np.unique(y_pred)):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    accuracy = cost_matrix[row_ind, col_ind].sum() / len(y_true)
    return accuracy

def kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_pred = kmeans.fit_predict(X)
    return y_pred

def make_blobs_dist(
    n_samples=100,
    n_features=2,
    centers=None,
    cluster_std=1.0,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=None,
    return_centers=False,
    dist = 't',
    df = 1,
):
    """
    Adapted from https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73/sklearn/datasets/_samples_generator.py#L917
    Generate t-distributed blobs for clustering.
    """
    generator = check_random_state(random_state)

    if isinstance(n_samples, numbers.Integral):
        # Set n_centers by looking at centers arg
        if centers is None:
            centers = 3

        if isinstance(centers, numbers.Integral):
            n_centers = centers
            centers = generator.uniform(
                center_box[0], center_box[1], size=(n_centers, n_features)
            )

        else:
            centers = check_array(centers)
            n_features = centers.shape[1]
            n_centers = centers.shape[0]

    else:
        # Set n_centers by looking at [n_samples] arg
        n_centers = len(n_samples)
        if centers is None:
            centers = generator.uniform(
                center_box[0], center_box[1], size=(n_centers, n_features)
            )
        if not isinstance(centers, Iterable):
            raise ValueError(
                "Parameter `centers` must be array-like. Got {!r} instead".format(
                    centers
                )
            )
        if len(centers) != n_centers:
            raise ValueError(
                "Length of `n_samples` not consistent with number of "
                f"centers. Got n_samples = {n_samples} and centers = {centers}"
            )
        centers = check_array(centers)
        n_features = centers.shape[1]

    # stds: if cluster_std is given as list, it must be consistent
    # with the n_centers
    if hasattr(cluster_std, "__len__") and len(cluster_std) != n_centers:
        raise ValueError(
            "Length of `clusters_std` not consistent with "
            "number of centers. Got centers = {} "
            "and cluster_std = {}".format(centers, cluster_std)
        )

    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.full(len(centers), cluster_std)

    if isinstance(n_samples, Iterable):
        n_samples_per_center = n_samples
    else:
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers

        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1

    cum_sum_n_samples = np.cumsum(n_samples_per_center)
    X = np.empty(shape=(sum(n_samples_per_center), n_features), dtype=np.float64)
    y = np.empty(shape=(sum(n_samples_per_center),), dtype=int)

    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        start_idx = cum_sum_n_samples[i - 1] if i > 0 else 0
        end_idx = cum_sum_n_samples[i]
        if dist == 't':
            sample = generator.standard_t(df=df, size=(n, n_features))
        elif dist == 'lognormal':
            sample = generator.lognormal(sigma=std, size=(n, n_features))
        else:
            sample = generator.normal(scale=std, size=(n, n_features))
        X[start_idx:end_idx] = centers[i] + sample
        y[start_idx:end_idx] = i

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if return_centers:
        return X, y, centers
    else:
        return X, y

def load_images_from_directory(base_directory, samples=5, clusters=2):
    images = []
    labels = []
    for i, label in enumerate(os.listdir(base_directory)[:clusters]):
        label_path = os.path.join(base_directory, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path)[:samples]:
                image_path = os.path.join(label_path, image_file)
                try:
                    image = Image.open(image_path)
                    images.append(image)
                    labels.append(i)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images[:], labels[:] = zip(*combined)
    return images, labels

def extract_list(text, pattern = r'\[([a-zA-Z0-9,\s]+)\]'):
    if isinstance(text, list):
        return text
    match = re.search(pattern, text)
    try:
        extracted_list = match.group(0)
        if not extracted_list.endswith(']'):
            extracted_list += ']'
        results = json.loads(extracted_list.strip())
    except json.JSONDecodeError:
        clean_list = extracted_list.strip("[]").split(",")
        results = [item.strip() for item in clean_list]
    except:
        results = []
    return results

def transform_label2num(y):
    le = LabelEncoder()
    le.fit(y)
    return le.transform(y)

def postprocess(text, num_samples):
    y_pred = transform_label2num(extract_list(text)).tolist()
    if len(y_pred) < num_samples:
        y_pred = y_pred + [-1]*(num_samples - len(y_pred))
    else:
        y_pred = y_pred[:num_samples]
    return y_pred

def extract_cluster_info(text):
    # Match the number of clusters
    num_clusters_match = re.search(r'(\d+)\s+clusters?', text)
    num_clusters = int(num_clusters_match.group(1)) if num_clusters_match else None
    # Match the condition
    condition_match = re.search(r'based\s+on\s+the\s+([\w\s]+)', text)
    condition = condition_match.group(1).strip().split()[0] if condition_match else None
    return num_clusters, condition

def get_random_class_sizes(n_samples, n_clusters):
    random_props = [random.random() for _ in range(n_clusters)]
    total_props = sum(random_props)
    remaining_samples = n_samples - n_clusters
    cluster_samples = [1+round(remaining_samples*p/total_props) for p in random_props] # make sure each cluster has at least one member
    cluster_samples[0] += n_samples - sum(cluster_samples) # allocate the remaining samples to the first cluster
    return cluster_samples

def move_to_cuda(data_dict):
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to('cuda')
        elif isinstance(value, list) and isinstance(value[0], torch.Tensor):
            data_dict[key] = [tensor.to('cuda') for tensor in value]
    return data_dict


def get_vision_chat_message(system_prompt, conversation):
    messages = []
    if system_prompt is not None:
        messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
        })
            
    for i, text in enumerate(conversation):
        if isinstance(text, dict): text = text['value']
        if i % 2 == 0:
            num_images = len([m.start() for m in re.finditer("<image>", text)])
            text = text.replace("<image>", "").strip()
            messages.append({
                "role": "user",
                        "content": [{"type": "text", "text": text}] + [{"type": "image"}] * num_images
            })
        else:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": text}]
            })
    return messages


def get_chat_message(system_prompt, conversation):
    messages = []
    if system_prompt is not None:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    for i, text in enumerate(conversation):
        if i % 2 == 0:
            messages.append({
                "role": "user",
                "content": text['value']
            })
        else:
            messages.append({
                "role": "assistant",
                "content": text['value']
            })
    return messages

def format_data(X, system_prompt, n_clusters, y=None):
    output = []
    output.append({'role': 'system', 'content': system_prompt})
    X_str = np.array2string(np.array(X).round(2), separator=',', suppress_small=True, precision=2).replace('\n', '')
    prompt = f"Cluster the following data into {n_clusters} clusters. Only output the cluster labels for each point as a list of integers. No code. "
    prompt += f"Data: \n{X_str.replace('[-', '[ -')}\nLabels:"
    output.append({'role': 'human', 'content': prompt})
    if y: 
        answer = np.array2string(np.array(y), separator=',').replace('\n', '').replace(' ', '').replace(',',', ')
        output.append({'role': 'human', 'content': answer})
    return output