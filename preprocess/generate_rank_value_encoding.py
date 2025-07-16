import argparse
import datetime
import os
import pickle
import random
import traceback
from pathlib import Path

import loompy
import numpy as np
import pandas as pd
import scanpy as sc
from datasets import Dataset
from loguru import logger

from geneformer.tokenizer import TOKEN_DICTIONARY_FILE, GENE_MEDIAN_FILE

parser = argparse.ArgumentParser(description="Scdifformer generate rank value encoding task")

parser.add_argument(
    '-i',
    '--input',
    required=True,
    type=str,
    help='The file path of h5ad. eg: /input/input.h5ad')
parser.add_argument(
    '-o',
    '--output',
    required=True,
    type=str,
    help='The output path. eg: /output')
parser.add_argument(
    '-gip',
    '--gene_info_path',
    default="/bgidata10T/scDifformer/data/pretain/material/gene_info_table.csv",
    type=str,
    help='The gene info path. eg: /input/gene_info_table.csv')
parser.add_argument(
    '-on',
    '--organ_name',
    type=str,
    default="pbmc",
    help='Organ name. eg: pbmc, prostate')
parser.add_argument(
    '-n',
    '--num_proc',
    type=int,
    default=8,
    help='The num proc to process dataset. eg: 8')

args = parser.parse_args()
input_path = args.input
output_path = args.output
gene_info_path = args.gene_info_path
organ_name = args.organ_name
num_proc = args.num_proc

gene_info = pd.read_csv(gene_info_path)
gene_name_id_combine_dict = gene_info.set_index("gene_name")["ensembl_id"].to_dict()
gene_name_type_dict = gene_info.set_index("gene_name")["gene_type"].to_dict()
gene_id_type_dict = gene_info.set_index("ensembl_id")["gene_type"].to_dict()
func_gene_list = [
    i
    for i in gene_info[
        (gene_info["gene_type"] == "protein_coding")
        | (gene_info["gene_type"] == "miRNA")
    ]["ensembl_id"]
]

with open(TOKEN_DICTIONARY_FILE, "rb") as f:
    gene_token_info = pickle.load(f)

def preflight():
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        logger.warning(f"{output_dir} not exists, creating it")
        os.makedirs(output_dir)


def preprocess(file_path: str) -> str:
    """
    Preprocess for h5ad, 如有必要可以加一些预处理
    """
    try:
        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        random_str = ''.join(random.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') for _ in range(8))
        task_id = date_str + random_str
        task_dir = os.path.dirname(output_path) + os.sep + task_id
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
        file_name = str(file_path).split(os.sep)[-1].split(".")[0]
        adata = sc.read_h5ad(file_path)
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        logger.info(f"file_path: {file_path} adata: {adata}")
        gene_type_arr = []
        ensembl_id_arr = []
        pretrain_uncover_gene_arr = []
        for var_info in adata.var_names:
            if (
                    var_info in gene_name_type_dict
                    and gene_name_id_combine_dict[var_info] in gene_token_info
            ):
                gene_type_arr.append(gene_name_type_dict[var_info])
                ensembl_id_arr.append(gene_name_id_combine_dict[var_info])
            elif str(var_info).startswith("ENSG") and var_info in gene_token_info:
                gene_type_arr.append(gene_id_type_dict[var_info])
                ensembl_id_arr.append(var_info)
            else:
                pretrain_uncover_gene_arr.append(var_info)
        if len(pretrain_uncover_gene_arr) != 0:
            logger.warning(f"Gene not in token file will be remove")
            adata = adata[:, adata.var_names.drop(pretrain_uncover_gene_arr)]
        if adata.n_obs == 0:
            raise Exception("Anndata.var_names should be gene name")
        total_genes_per_row = adata.X.sum(axis=1)
        filename = f"{task_dir}/{file_name}.loom"
        loompy.create(
            filename,
            adata.X.T,
            {
                "ensembl_id": np.array(ensembl_id_arr),
                "gene_type": gene_type_arr,
            },
            {"n_counts": total_genes_per_row, "barcode": np.array(adata.obs_names), "cell_type": np.array(adata.obs["cell_type"])},
        )
        logger.info(f"{filename} is done")
        del adata
        return task_dir
    except Exception as e:
        logger.error(f"====={file_path} {e}")
        logger.error(f"{traceback.print_exc()}")
        raise e


def tokenize_cell(gene_vector, gene_tokens):
    """
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    # create array of gene vector with token indices
    # mask undetected genes
    nonzero_mask = np.nonzero(gene_vector)[0]
    # sort by median-scaled gene values
    sorted_indices = np.argsort(-gene_vector[nonzero_mask])
    # tokenize
    sentence_tokens = gene_tokens[nonzero_mask][sorted_indices]
    return sentence_tokens


class TranscriptomeTokenizer:
    def __init__(
        self,
        custom_attr_name_dict,
        nproc=1,
        gene_median_file=GENE_MEDIAN_FILE,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
    ):
        """
        Initialize tokenizer.
        Parameters
        ----------
        custom_attr_name_dict : dict
            Dictionary of custom attributes to be added to the dataset.
            Keys are the names of the attributes in the loom file.
            Values are the names of the attributes in the dataset.
        nproc : int
            Number of processes to use for dataset mapping.
        gene_median_file : Path
            Path to pickle file containing dictionary of non-zero median
            gene expression values across Genecorpus-30M.
        token_dictionary_file : Path
            Path to pickle file containing token dictionary (Ensembl IDs:token).
        """
        # dictionary of custom attributes {output dataset column name: input .loom column name}
        self.custom_attr_name_dict = custom_attr_name_dict

        # number of processes for dataset mapping
        self.nproc = nproc

        # load dictionary of gene normalization factors
        # (non-zero median value of expression across Genecorpus-30M)
        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        # gene keys for full vocabulary
        self.gene_keys = list(self.gene_median_dict.keys())

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

    def tokenize_data(self, loom_data_directory, output_directory, output_prefix):
        """
        Tokenize .loom files in loom_data_directory and save as tokenized .dataset in output_directory.
        Parameters
        ----------
        loom_data_directory : Path
            Path to directory containing loom files
        output_directory : Path
            Path to directory where tokenized data will be saved as .dataset
        output_prefix : str
            Prefix for output .dataset
        """
        tokenized_cells, cell_metadata = self.tokenize_files(Path(loom_data_directory))
        tokenized_dataset = self.create_dataset(tokenized_cells, cell_metadata)

        output_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
        tokenized_dataset.save_to_disk(output_path, max_shard_size="5000MB")

    def tokenize_files(self, loom_data_directory):
        tokenized_cells = []
        loom_cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
        cell_metadata = {
            attr_key: [] for attr_key in self.custom_attr_name_dict.values()
        }
        i = 1
        # loops through directories to tokenize .loom files
        for loom_file_path in loom_data_directory.glob("*.loom"):
            i += 1
            file_tokenized_cells, file_cell_metadata = self.tokenize_file(
                loom_file_path
            )
            tokenized_cells += file_tokenized_cells
            for k in loom_cell_attr:
                cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[k]

        return tokenized_cells, cell_metadata

    def tokenize_file(self, loom_file_path):
        file_cell_metadata = {
            attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
        }
        with loompy.connect(str(loom_file_path)) as data:
            # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors
            coding_miRNA_loc = np.where(
                [self.genelist_dict.get(i, False) for i in data.ra["ensembl_id"]]
            )[0]
            norm_factor_vector = np.array(
                [
                    self.gene_median_dict[i]
                    for i in data.ra["ensembl_id"][coding_miRNA_loc]
                ]
            )
            coding_miRNA_ids = data.ra["ensembl_id"][coding_miRNA_loc]
            coding_miRNA_tokens = np.array(
                [self.gene_token_dict[i] for i in coding_miRNA_ids]
            )

            # define coordinates of cells passing filters for inclusion (e.g. QC)
            try:
                data.ca["filter_pass"]
            except AttributeError:
                var_exists = False
            else:
                var_exists = True

            if var_exists is True:
                filter_pass_loc = np.where(
                    [True if i == 1 else False for i in data.ca["filter_pass"]]
                )[0]
            elif var_exists is False:
                print(
                    f"{loom_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
                )
                filter_pass_loc = np.array([i for i in range(data.shape[1])])

            # scan through .loom files and tokenize cells
            tokenized_cells = []
            for _ix, _selection, view in data.scan(items=filter_pass_loc, axis=1):
                # select subview with protein-coding and miRNA genes
                subview = view.view[coding_miRNA_loc, :]

                # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
                # and normalize by gene normalization factors
                subview_norm_array = (
                    subview[:, :]
                    / subview.ca.n_counts
                    * 10_000
                    / norm_factor_vector[:, None]
                )
                # tokenize subview gene vectors
                tokenized_cells += [
                    tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens)
                    for i in range(subview_norm_array.shape[1])
                ]

                # add custom attributes for subview to dict
                for k in file_cell_metadata.keys():
                    file_cell_metadata[k] += subview.ca[k].tolist()

        return tokenized_cells, file_cell_metadata

    def create_dataset(self, tokenized_cells, cell_metadata):
        # create dict for dataset creation
        dataset_dict = {"input_ids": tokenized_cells}
        dataset_dict.update(cell_metadata)

        # create dataset
        output_dataset = Dataset.from_dict(dataset_dict)

        # truncate dataset
        def truncate(example):
            example["input_ids"] = example["input_ids"][0:2048]
            return example

        output_dataset_truncated = output_dataset.map(truncate, num_proc=self.nproc)

        # measure lengths of dataset
        def measure_length(example):
            example["length"] = len(example["input_ids"])
            return example

        output_dataset_truncated_w_length = output_dataset_truncated.map(
            measure_length, num_proc=self.nproc
        )

        return output_dataset_truncated_w_length


def main():
    preflight()
    task_dir = preprocess(file_path=input_path)
    tk = TranscriptomeTokenizer(custom_attr_name_dict={"barcode": "barcode", "cell_type": "cell_type"})
    # tk = TranscriptomeTokenizer(custom_attr_name_dict={"barcode": "barcode"})
    output_prefix="cell_type_anno"
    logger.info("Tokenize data start")
    tk.tokenize_data(task_dir, task_dir, output_prefix)
    logger.info("Tokenize data end")
    rank_value_encoding_path = os.path.join(task_dir, output_prefix + ".dataset")
    logger.info(f"Rank value encoding done: {rank_value_encoding_path}")

if __name__ == '__main__':
    main()
