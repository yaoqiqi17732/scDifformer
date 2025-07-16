"""Extra cell embedding"""
import os
import traceback
from multiprocessing import Pool, cpu_count
from typing import Dict, Set, Tuple
import pandas as pd
import scanpy as sc
from loguru import logger
from datetime import datetime
import pickle
from pathlib import Path
from transformers import set_seed
import loompy
import numpy as np
import scrublet
from collections import Counter
from datasets import Dataset, load_from_disk
from geneformer import EmbExtractor
set_seed(12)

current_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")

# Add log file
current_file_name = os.path.splitext(os.path.basename(__file__))[0]
logger.add(f"./log/{current_file_name}.log")

# Common parm
num_processes = 8
forward_batch_size = 32
cell_type_flag = True
qc_filter_flag = True

# Work dir
home_path ="/bgidata10T/scDifformer/code/cross_species"
input_dir = f"{home_path}/mus/input"
output_dir = f"{home_path}/mus/output/{current_time}"

# Gene mapping parm
human_rbh_mouse_path = f"{home_path}/material/human-rbh-mouse.xlsx"
gene_mapping_h5ad_dir = f"{output_dir}/gene_mapping_output"

# Preprocess parm
gene_info_table_path = f"{home_path}/material/gene_info_table.csv"
loom_dir = f"{output_dir}/looms"
input_dir_h5ad = gene_mapping_h5ad_dir

# Tokenizer parm
GENE_MEDIAN_FILE = f"{home_path}/material/mus_gene_median_dictionary.pkl"
TOKEN_DICTIONARY_FILE = f'{home_path}/material/token_dictionary.pkl'
output_directory = output_dir
output_prefix = f"rve_run_{current_time}"

# Cell embedding parm
rank_value_encoding_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
pretrain_model_path = '/bgidata/scDifformer/models/L24.models'

logger.info(f"current_time: {current_time} num_processes: {num_processes} qc_filter_flag: {qc_filter_flag}")

gene_info = pd.read_csv(gene_info_table_path)

# Convert gene info table to dict 
gene_name_id_combine_dict = gene_info.set_index('gene_name')['ensembl_id'].to_dict()
gene_name_type_dict = gene_info.set_index('gene_name')['gene_type'].to_dict()
gene_id_type_dict = gene_info.set_index('ensembl_id')['gene_type'].to_dict()
func_gene_list = [i for i in gene_info[(gene_info["gene_type"] == "protein_coding") | (gene_info["gene_type"] == "miRNA")]["ensembl_id"]]

gene_token_info = {}
with open(TOKEN_DICTIONARY_FILE, "rb") as f:
    gene_token_info = pickle.load(f)

def basic_check() -> None:
    """
    Basic check and create necessary work dir
    """
    if not os.path.exists(input_dir):
        logger.warning(f"{input_dir} does not exits! make it.")
        os.makedirs(input_dir)
    if not os.path.exists(gene_mapping_h5ad_dir):
        logger.info(f"{gene_mapping_h5ad_dir} does not exits! make it.")
        os.makedirs(gene_mapping_h5ad_dir)
    else:
        logger.warning(f"{input_dir} already exists, pls ensure there are no duplicates.")
    if not os.path.exists(loom_dir):
        logger.info(f"{loom_dir} does not exits! make it.")
        os.makedirs(loom_dir)

def load_mapping_data() -> Tuple[Dict[str, str], Set[str]]:
    """
    Read human and mus gene info
    """
    df_mapping = pd.read_excel(human_rbh_mouse_path)
    df_mapping.set_index("Mus_Gene_Name", inplace=True)
    mus_human_dict = df_mapping["Human_Gene_Name"].to_dict()
    mus_gene_name_id_dict = df_mapping["Mus_Gene_ID"].to_dict()
    return mus_human_dict, set(mus_gene_name_id_dict.keys())


def process_gene_mapping(file_path: str) -> None:
    """
    Process gene mapping for a given h5ad file.
    """
    adata = sc.read_h5ad(file_path)
    if "feature_name" not in adata.var:
        logger.error(f"`feature_name` must be in adata.var for {file_path}")
        raise ValueError(f"`feature_name` must be in adata.var for {file_path}")
    logger.info(f"{file_path} processed start n_cell * n_gene: {adata.n_obs} * {adata.n_vars}.")
    try:
        mus_human_gene_dict, source_gene_name_set = load_mapping_data()
        adata.var_names = list(adata.var["feature_name"])
        retain_gene_set = source_gene_name_set.intersection(adata.var_names)
        adata = adata[:, list(retain_gene_set)]
        adata.var["mapping_human_feature_name"] = adata.var.index.map(mus_human_gene_dict)
        adata = adata[:, ~adata.var["mapping_human_feature_name"].isna()]
        adata.var_names = list(adata.var["mapping_human_feature_name"])
        adata.var_names_make_unique()
        file_name = os.path.basename(file_path)
        h5ad_save_path = os.path.join(gene_mapping_h5ad_dir, file_name)
        adata.write_h5ad(h5ad_save_path)
        logger.info(f"{file_path} processed successfully.")
        del adata
    except Exception as e:
        logger.error(f'Error processing {file_path} {traceback.print_exc()}')
        raise Exception(f'Error processing {file_path} {traceback.print_exc()}')

def preprocess(file_path: str) -> None:
    """
    Preprocess for h5ad
    """
    try:
        file_name = file_path.split(os.sep)[-1].split(".")[0]
        adata = sc.read_h5ad(file_path)
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        if qc_filter_flag:
            # 1. Remove mitochondria and total number of genes within 3 s.d. of the mean
            adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
            sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

            # Calculate the mean and standard deviation of mitochondrial proportion
            mean_ratio_mt = np.mean(adata.obs["pct_counts_mt"])
            std_ratio_mt = np.std(adata.obs["pct_counts_mt"])

            # Calculate the cells within three standard deviations of the mean
            min_ratio_mt = mean_ratio_mt - 3 * std_ratio_mt
            max_ratio_mt = mean_ratio_mt + 3 * std_ratio_mt
            within_range_mt = (adata.obs["pct_counts_mt"] >= min_ratio_mt) & (adata.obs["pct_counts_mt"] <= max_ratio_mt)
            adata = adata[within_range_mt, :]

            # Calculate the mean and standard deviation of the total expression level of line genes
            mean_ratio = np.mean(adata.obs["total_counts"])
            std_ratio = np.std(adata.obs["total_counts"])

            # Calculate the cells within three standard deviations of the mean
            min_ratio = mean_ratio - 3 * std_ratio
            max_ratio = mean_ratio + 3 * std_ratio
            within_range = (adata.obs["total_counts"] >= min_ratio) & (adata.obs["total_counts"] <= max_ratio)
            adata = adata[within_range]

            # 2. Create DoubletFinder object and run bimodal cell detection
            scrub = scrublet.Scrublet(adata.X)
            # Run doublet detection
            _, predicted_doublets = scrub.scrub_doublets()
            # Filter the cells based on the doublet scores and predicted doublets
            if predicted_doublets is not None:
                adata = adata[~predicted_doublets]

            # 3. exclude cells with less than seven detected Ensembl-annotated protein-coding or miRNA genes
            retain_gene_set = set()
            func_gene_set = set(func_gene_list)
            for gene_name in adata.var_names:
                if gene_name in func_gene_set:
                    retain_gene_set.add(gene_name)
                elif gene_name in gene_name_id_combine_dict and gene_name_id_combine_dict[gene_name] in func_gene_set:
                    retain_gene_set.add(gene_name)

            adata = adata[:, list(retain_gene_set)]
            sc.pp.filter_cells(adata, min_genes=7)
        gene_type_arr = []
        ensembl_id_arr = []
        pretain_uncover_gene_arr = []
        for var_info in adata.var_names:
            if var_info in gene_name_type_dict and gene_name_id_combine_dict[var_info] in gene_token_info:
                gene_type_arr.append(gene_name_type_dict[var_info])
                ensembl_id_arr.append(gene_name_id_combine_dict[var_info])
            elif str(var_info).startswith("ENSG") and var_info in gene_token_info:
                gene_type_arr.append(gene_id_type_dict[var_info])
                ensembl_id_arr.append(var_info)
            else:
                pretain_uncover_gene_arr.append(var_info)
        if len(pretain_uncover_gene_arr) != 0:
            uncover_gene_info = ', '.join(map(str, pretain_uncover_gene_arr))
            logger.warning(f"Gene not in token file will remove: {uncover_gene_info}")
            # raise
            adata = adata[:, adata.var_names.drop(pretain_uncover_gene_arr)]

        total_genes_per_row = adata.X.sum(axis=1)
        filename = f'{loom_dir}/{file_name}.loom'
        if "cell_type" in adata.obs:
            loompy.create(filename, adata.X.T, {"ensembl_id": np.array(ensembl_id_arr), "gene_type": gene_type_arr,},
                      {"n_counts": total_genes_per_row, "cell_type": np.array(adata.obs["cell_type"]), "barcode": np.array(adata.obs_names)})
        else:
            loompy.create(filename, adata.X.T, {"ensembl_id": np.array(ensembl_id_arr), "gene_type": gene_type_arr,},
                      {"n_counts": total_genes_per_row, "barcode": np.array(adata.obs_names)})            
        logger.info(f"{filename} is done")
        del adata
    except Exception as e:
        logger.error(f"====={file_path} {e}")
        logger.error(f'{traceback.print_exc()}')

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
        tokenized_dataset.save_to_disk(output_path, max_shard_size='500000MB')

    def tokenize_files(self, loom_data_directory):
        tokenized_cells = []
        loom_cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
        cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}
        i = 1
        # loops through directories to tokenize .loom files
        for loom_file_path in loom_data_directory.glob("*.loom"):
            logger.info(f"已经处理的文件个数 {i}")
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
            for (_ix, _selection, view) in data.scan(items=filter_pass_loc, axis=1):
                # select subview with protein-coding and miRNA genes
                subview = view.view[coding_miRNA_loc, :]

                # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
                # and normalize by gene normalization factors
                tmp = subview[:, :] / subview.ca.n_counts * 10_000
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
    
def cell_embedding_pretain() -> None:
    ds = load_from_disk(rank_value_encoding_path)
    logger.info(f"Ds load success! {ds}")
    if 'cell_type' in ds.features:
        target_names = list(Counter(ds["cell_type"]).keys())

        embex = EmbExtractor(model_type="Pretrained",
                            num_classes=0,
                            filter_data={"cell_type": target_names},
                            max_ncells=ds.num_rows,
                            emb_layer=-1,
                            emb_label=["cell_type", "barcode"],
                            labels_to_plot=["cell_type", "barcode"],
                            forward_batch_size=forward_batch_size,
                            nproc=num_processes)

        embex.extract_embs(pretrain_model_path,
                                rank_value_encoding_path,
                                output_directory,
                                f"pretain_cell_emb_celltype_barcode_{current_time}")
    else:
        embex = EmbExtractor(model_type="Pretrained",
                            num_classes=0,
                            max_ncells=ds.num_rows,
                            emb_layer=-1,
                            emb_label=["barcode"],
                            labels_to_plot=["barcode"],
                            forward_batch_size=forward_batch_size,
                            nproc=num_processes)

        embex.extract_embs(pretrain_model_path,
                                rank_value_encoding_path,
                                output_directory,
                                f"pretain_cell_emb_barcode_{current_time}")


def cell_embedding_cellclassifier() -> None:
    ds = load_from_disk(rank_value_encoding_path)
    logger.info(f"Ds load success! {ds}")
    if 'cell_type' in ds.features:
        target_names = list(Counter(ds["cell_type"]).keys())
        num_celltype=len(target_names)

        embex = EmbExtractor(model_type="CellClassifier",
                            num_classes=num_celltype,
                            filter_data={"cell_type": target_names},
                            max_ncells=ds.num_rows,
                            emb_layer=-1,
                            emb_label=["cell_type", "barcode"],
                            labels_to_plot=["cell_type", "barcode"],
                            forward_batch_size=forward_batch_size,
                            nproc=num_processes)

        embex.extract_embs(pretrain_model_path,
                                rank_value_encoding_path,
                                output_directory,
                                f"cellclassifier_cell_emb_celltype_barcode_{current_time}")
    else:
        embex = EmbExtractor(model_type="CellClassifier",
                            num_classes=0,
                            max_ncells=ds.num_rows,
                            emb_layer=-1,
                            emb_label=["barcode"],
                            labels_to_plot=["barcode"],
                            forward_batch_size=forward_batch_size,
                            nproc=num_processes)

        embex.extract_embs(pretrain_model_path,
                                rank_value_encoding_path,
                                output_directory,
                                f"cellclassifier_cell_emb_barcode_{current_time}")

if __name__ == '__main__':
    basic_check()

    # Gene mapping start
    gene_mapping_file_paths = []
    for file in os.listdir(input_dir):
        if not file.endswith(".h5ad"):
            continue
        gene_mapping_file_paths.append(os.path.join(input_dir, file))
    work_len = len(gene_mapping_file_paths)
    if work_len == 0:
        logger.error("Input dir is empty")
        raise
    logger.info(f"Gene mapping work start and file length: {work_len}")

    num_processes = min(num_processes, cpu_count())
    with Pool(num_processes) as pool:
        pool.map(process_gene_mapping, gene_mapping_file_paths)

    logger.info(f"Gene mapping work end.")
    # Gene mapping end

    # Preprocess start
    files = [os.path.join(input_dir_h5ad, file_name) for file_name in os.listdir(input_dir_h5ad) if str(file_name).endswith(".h5ad")]
    logger.info(f'Preprocess work start and file length: {len(files)}')
    with Pool(num_processes) as pool:
        pool.map(preprocess, files)    
    logger.info(f'Preprocess work end')
    # Preprocess end 

    # Tokenize start
    logger.info(f'Tokenize work start')
    if cell_type_flag:
         tk = TranscriptomeTokenizer(custom_attr_name_dict={"cell_type": "cell_type", "barcode": "barcode"}, nproc=num_processes)
    else:
         tk = TranscriptomeTokenizer(custom_attr_name_dict={"barcode": "barcode"}, nproc=num_processes)
    tk.tokenize_data(loom_dir, output_directory, output_prefix)
    logger.info(f'Tokenize work end')
    # Tokenize end

    # Cell embedding start
    logger.info(f'Cell embedding start')
    cell_embedding_pretain()
    cell_embedding_cellclassifier()
    logger.info(f'Cell embedding end')
    # Cell embedding end
