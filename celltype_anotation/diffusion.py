import os
import scanpy as sc
import numpy as np
import random
from sklearn.model_selection import KFold
import sys
import dill
import torch
import anndata as ad
from scipy import sparse
import pandas as pd
import geneformer
from collections import OrderedDict
from collections import Counter
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
print(geneformer.__file__)

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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_number', type=int, default=0, help='the number of gpu to be used')
parser.add_argument('--fold_count', type=int, default=5, help='The fold number need to be trained')
parser.add_argument('--data_path', type=str, default='/mnt/nfs/wbzhang/data/celltype_annotation/zheng68k_final/data/geneformer/NEW_5fold/', help='The arrow data need to be trained')
parser.add_argument('--pretrain_model_path', type=str, default='/mnt/nfs/xz/diffusion/db/Diffusion-BERT/model_out/231004_000713_model_name_gene_lr_1e-08_seed_42_numsteps_256_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_False_timestep_none_ckpts/best_0_14999.th', help='The pretrained model')
parser.add_argument('--output_path', type=str, default='/mnt/nfs/wbzhang/data/celltype_annotation/zheng68k_final/Model/diffusion/24L_256_batch6/', help='the output path')
args = parser.parse_args()

data_path = args.data_path
output_path = args.output_path
pretrain_model_path = args.pretrain_model_path
fold_count = args.fold_count

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

# set model parameters
# max input size
max_input_size = 2 ** 11  # 2048

# set training hyperparameters
# max learning rate
max_lr = 5e-5
# how many pretrained layers to freeze
freeze_layers = 0
# number gpus
num_gpus = 1
# number cpu cores
num_proc = 16
# batch size for training and eval
geneformer_batch_size = 6
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 500
# number of epochs
epochs = 10
# optimizer
optimizer = "adamw"

for i in range(fold_count):
    with open(data_path + 'fold' + str(i) + '_train.pkl', 'rb') as f:
        trainset = dill.load(f)
    with open(data_path + 'name_id.pkl', 'rb') as f:
        name_id = dill.load(f)

    train_split = trainset.select([i for i in range(0,round(len(trainset)*0.9))])
    eval_split = trainset.select([i for i in range(round(len(trainset)*0.9),len(trainset))])
    
    # set logging steps
    logging_steps = round(len(train_split)/geneformer_batch_size/10)

    #reload pretrained model
    model_NEW_24L = '/mnt/nfs/xz/geneformer/Geneformer34m/examples/pretraining_new_model/output_own/models/230814_102419_geneformer_34M_L24_emb512_SL2048_E5_B6_LR0.0003_LSlinear_WU20000_Oadamw_DS64/models'
    model = BertForSequenceClassification.from_pretrained(model_NEW_24L,
                                                      num_labels=len(name_id.keys()),
                                                      output_attentions = False,
                                                      output_hidden_states = False).to("cuda")
    checkpoint = torch.load(pretrain_model_path)

    new_state_dict = OrderedDict()
    for key, v in checkpoint['model'].items():
        name = key.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict,strict=False)

    output_dir = output_path + 'fold' + str(i) + '_model'
    print(output_dir)
    saved_model_test = output_dir + '/pytorch_model.bin'
    
    if os.path.isfile(saved_model_test) == True:
        raise Exception("Model already saved to this directory.")
    # make output directory
    subprocess.call(f'mkdir {output_dir}', shell=True)

    # set training arguments
    training_args = {
        "learning_rate": max_lr,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "load_best_model_at_end": True,
        "output_dir": output_dir,
    }
    
    training_args_init = TrainingArguments(**training_args)

    trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=train_split,
        eval_dataset=eval_split,
        compute_metrics=compute_metrics
    )
    
    # train the cell type classifier
    trainer.train()
    predictions = trainer.predict(eval_split)
    with open(f"{output_dir}predictions.pickle", "wb") as fp:
        pickle.dump(predictions, fp)
    trainer.save_metrics("eval",predictions.metrics)
    trainer.save_model(output_dir)
