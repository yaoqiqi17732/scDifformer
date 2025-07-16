import os
import scanpy as sc
import numpy as np
import random
from sklearn.model_selection import KFold
import sys
import dill
import torch
import time
import anndata as ad
from scipy import sparse
import pandas as pd
import geneformer
from collections import Counter
from datasets import load_from_disk
from collections import OrderedDict
from sklearn.metrics import accuracy_score, f1_score
print(geneformer.__file__)
from functools import reduce

from transformers import BertForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments
from geneformer import DataCollatorForCellClassification
import subprocess
import pickle

import seaborn as sns; sns.set()
from geneformer import TranscriptomeTokenizer
import argparse

GPU_NUMBER = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"

from geneformer.in_silico_perturber import downsample_and_sort, \
                                 gen_attention_mask, \
                                 get_model_input_size, \
                                 load_and_filter, \
                                 load_model, \
                                 pad_tensor_list, \
                                 quant_layers, \
                                 pad_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--layer_number', type=int, default=-24, help='the layer number to be extract')
args = parser.parse_args()
layer_number = args.layer_number

fold_path = '/mnt/nfs/wbzhang/data/celltype_annotation/lung/1/data/geneformer/tokenized_lung_1.dataset'
dataset = load_from_disk(fold_path)
target_names = list(Counter(dataset["cell_type"]).keys())
target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))

dataset.set_format(type='torch')

model_finetune_path = '/mnt/nfs/wbzhang/data/celltype_annotation/lung/1/Model/diffusion_24L_2048/'
model_finetune_dir = model_finetune_path + 'fold0_model'

model = BertForSequenceClassification.from_pretrained(model_finetune_dir, 
                                              num_labels=len(target_names),
                                              output_attentions = True,
                                              output_hidden_states = True).to("cuda")

from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
token_dictionary_file=TOKEN_DICTIONARY_FILE

with open(token_dictionary_file, "rb") as f:
    gene_token_dict = pickle.load(f)
pad_token_id = gene_token_dict.get("<pad>")
# 取出对应token的gene

ref_keys = list(gene_token_dict.keys())[2:]

random.seed(2023)

for cell_type in target_names:
    df_attention = pd.DataFrame(columns=list(gene_token_dict.keys())[2:])

    total_list = list(np.arange(len(dataset))[(np.array(dataset['cell_type']) == cell_type)])
    target_set = dataset[random.sample(total_list, len(total_list) // 10)]

    input_ids = target_set['input_ids']
    length = target_set['length']
    print(len(length))
    for i in range(len(length)):
        input_data = input_ids[i:(i+1)]
        model_input_size = length[i:(i+1)]
        input_data_minibatch = pad_tensor_list(input_data, 
                                           model_input_size, 
                                           pad_token_id, 
                                           model_input_size)
        with torch.no_grad():
            outputs = model(
                input_ids = input_data_minibatch.to('cuda')
            )# attention mask 是为了标记padding，单个句子可以不用管
        attentions = outputs['attentions']
        # 对attention 按照scbert的方式进行处理
        attention = torch.sum(torch.mean(attentions[layer_number].squeeze(0), dim=0), dim=0)
        input_values = list(input_data[0].cpu().numpy()) #输入cell的值
        index_input_values = list(map(lambda x: x-2, input_values))
        gene_names = list(np.array(ref_keys)[index_input_values])

        temp_array = np.zeros(len(ref_keys))
        temp_array[index_input_values] = attention.cpu().numpy()
        df_attention.loc[i] = temp_array
        
        del outputs
        del input_data_minibatch
        del attentions
        del attention
        del input_values
        del index_input_values
        del gene_names
        del temp_array
        torch.cuda.empty_cache()
    df_attention['cell_type'] = [cell_type] * len(length)
    type_number = target_name_id_dict[cell_type]
    output_path='/mnt/nfs/wbzhang/data/celltype_annotation/analysis/data_lung1_attention/10%_finetune/'
    os.makedirs(output_path + str(layer_number) + 'layer/' + 'type' + str(type_number), exist_ok=True)
    
    df_attention.to_csv(output_path + str(layer_number) + 'layer/' + 'type' + str(type_number) + '/attention_weights.csv')