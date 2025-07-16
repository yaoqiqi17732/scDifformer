import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from pandas import Series,DataFrame
import geneformer
import pickle
print(geneformer.tokenizer.TOKEN_DICTIONARY_FILE)
with open(geneformer.tokenizer.TOKEN_DICTIONARY_FILE, 'rb') as fp:
    a = pickle.load(fp)
print(len(a))

import dill
from datasets import load_from_disk
from transformers import Trainer
from transformers import BertForSequenceClassification
from geneformer import DataCollatorForCellClassification
import torch
import random
#random.seed(2023)
from sklearn.metrics import accuracy_score, f1_score
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from geneformer.in_silico_perturber import make_perturbation_batch, \
                                           overexpress_tokens, \
                                           delete_indices, \
                                           get_model_input_size, \
                                           pad_or_truncate_encoding, \
                                           measure_length

# 读取数据
data_path_20_30 = '/mnt/nfs/wbzhang/data/perturber/aging/data/tokenized_data/20_30/tokenized_aging.dataset'
data_path_60_80 = '/mnt/nfs/wbzhang/data/perturber/aging/data/tokenized_data/60_80/tokenized_aging.dataset'
dataset_20_30 = load_from_disk(data_path_20_30)
dataset_60_80 = load_from_disk(data_path_60_80)
# 选Female数据
dataset_68F = dataset_60_80.select(list(np.arange(len(dataset_60_80))[np.array(dataset_60_80['gender']) == 'F']))
dataset_23F = dataset_20_30.select(list(np.arange(len(dataset_20_30))[np.array(dataset_20_30['gender']) == 'F']))
# 选择预测差距偏小的
pred_60_80 = pd.read_csv('/mnt/nfs/wbzhang/data/perturber/aging/data/60_80_F_predictions.csv',sep=',') 
testset_60_80 = dataset_68F.select(list(np.arange(len(dataset_68F))[(np.array((abs(pred_60_80['label'] - pred_60_80['original_predictions'])) <= 0.5))]))
# 只选择60岁
testset_60_80 = testset_60_80.select(list(np.arange(len(testset_60_80))[np.array(testset_60_80['age']) == 60]))
testset_60_80 = testset_60_80.rename_column('age', 'label')
print('final choose of 60_80:', len(testset_60_80))

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
      'accuracy': acc,
      'macro_f1': macro_f1
    }

def pad_or_truncate_encoding_max(encoding, pad_token_id, max_len):
    if isinstance(encoding, torch.Tensor):
        encoding_len = tensor.size()[0]
    elif isinstance(encoding, list):
        encoding_len = len(encoding)
    if encoding_len > max_len:
        encoding = encoding[0:max_len]

    return encoding

model_finetune_path = '/mnt/nfs/xz/aging/top6cell/regression_model/all/fold0_model/231130_0041_age_L2048_B5_LR1e-05_LSlinear_WU500_E4_Oadamw_F0/'
model = BertForSequenceClassification.from_pretrained(model_finetune_path, 
                                                      num_labels=1,
                                                      output_attentions = False,
                                                      output_hidden_states = False).to('cuda')
trainer = Trainer(
model=model,
data_collator=DataCollatorForCellClassification(),
compute_metrics=compute_metrics
)
original_predictions = trainer.predict(testset_60_80)
original_predictions = original_predictions.predictions

token_dictionary_file=TOKEN_DICTIONARY_FILE
perturb_type = 'overexpress'
model_input_size = get_model_input_size(model)
nproc=16

with open(token_dictionary_file, "rb") as f:
    gene_token_dict = pickle.load(f)
pad_token_id = gene_token_dict.get("<pad>")

df_based = pd.read_csv('/mnt/nfs/wbzhang/data/perturber/aging/data/results/ranking_version/OCT4_SOX2_KLF4_choose1_ranking.csv',sep=',',index_col=0)
top20_genes = list(df_based.index)[:20]
count = 0
for i in range(20)[:18]:
    gene1 = top20_genes[i]
    for j in range(20)[i+1:19]:
        gene2 = top20_genes[j]
        for k in range(20)[j+1:]:
            count += 1
            print('Already count for:', count)
            gene3 = top20_genes[k]
            genes_to_perturb=[gene1, gene2, gene3]
            missing_genes = [gene for gene in genes_to_perturb if gene not in gene_token_dict.keys()]
            if len(missing_genes) == len(genes_to_perturb):
                logger.error(
                    "None of the provided genes to perturb are in token dictionary."
                )
                raise
            elif len(missing_genes)>0:
                logger.warning(
                    f"Genes to perturb {missing_genes} are not in token dictionary.")
            tokens_to_perturb = [gene_token_dict.get(gene) for gene in genes_to_perturb]
            def make_group_perturbation_batch(example):
                example_input_ids = example["input_ids"]
                example["tokens_to_perturb"] = tokens_to_perturb
                indices_to_perturb = [example_input_ids.index(token) if token in example_input_ids else None for token in tokens_to_perturb]
                indices_to_perturb = [item for item in indices_to_perturb if item is not None]
                if len(indices_to_perturb) > 0:
                    example["perturb_index"] = indices_to_perturb
                else:
                    # -100 indicates tokens to overexpress are not present in rank value encoding
                    example["perturb_index"] = [-100] #如果没有出现就是-100
                if perturb_type == "delete":
                    example = delete_indices(example)
                elif perturb_type == "overexpress":
                    example = overexpress_tokens(example) 
                return example
            perturbation_batch = testset_60_80.map(make_group_perturbation_batch, num_proc=nproc)
            
            model_input_size = get_model_input_size(model)
            length_set = set(perturbation_batch['length'])
            lengths = perturbation_batch['length']
            if (len(length_set) > 1) or (max(length_set) > model_input_size):
                needs_pad_or_trunc = True #需要将input size length对齐
            else:
                needs_pad_or_trunc = False
                max_len = max(minibatch_length_set)
            
            if needs_pad_or_trunc == True:
                max_len = min(max(length_set),model_input_size) #最长只能到2048
                def pad_or_trunc_example(example):
                    # 此处有修改 我们把长的截断，把短的留下来
                    example["input_ids"] = pad_or_truncate_encoding_max(example["input_ids"], 
                                                                        pad_token_id, 
                                                                        max_len)   
                    return example
            perturbation_batch = perturbation_batch.map(pad_or_trunc_example, num_proc=nproc)
            perturbation_batch = perturbation_batch.map(
                    measure_length, num_proc=nproc
                )    
            perturb_predictions = trainer.predict(perturbation_batch)
            perturb_predictions = perturb_predictions.predictions
            data_output = {
                'label': Series(testset_60_80['label']),
                'original_predictions': Series(list(original_predictions.squeeze())),
                'perturb_predictions': Series(list(perturb_predictions.squeeze()))
            }
            df_output = DataFrame(data_output)
            df_output.to_csv('/mnt/nfs/wbzhang/data/perturber/aging/data/top20genes_choose3/'+ gene1 + '_' + gene2 + '_' + gene3 + '.csv', index=False, header=True)
            del perturbation_batch
            torch.cuda.empty_cache()