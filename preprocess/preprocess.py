"""
Single-cell data preprocessing for ScDifformer Model.
"""

from __future__ import annotations

import argparse
import datetime
import pickle
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Union

import loompy
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from datasets import Dataset
from loguru import logger
from scrublet import Scrublet


def find_project_root(marker_file: str = "pyproject.toml") -> Path:
    """Locate the project root directory by searching for a marker file.

    Args:
        marker_file: File name that identifies the project root.

    Returns:
        Path to the project root directory.

    Raises:
        FileNotFoundError: If marker file is not found in any parent directory.
    """
    current_dir = Path(__file__).resolve().parent

    while True:
        if (current_dir / marker_file).exists():
            return current_dir

        parent_dir = current_dir.parent

        if parent_dir == current_dir:
            raise FileNotFoundError(f"Could not find {marker_file} in any parent directory")

        current_dir = parent_dir


sys.path.insert(0, str(find_project_root()))

from libs import (
    GENE_INFO_PATH,
    GENE_MEDIAN_FILE,
    TOKEN_DICTIONARY_FILE,
    DEFAULT_OUTPUT_DIR,
    EXAMPLE_DATA_DIR,
)  # noqa


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the processing pipeline."""
    parser = argparse.ArgumentParser(description="ScDifformer generate rank value encoding task")

    parser.add_argument(
        "-i",
        "--input",
        required=False,
        default=EXAMPLE_DATA_DIR,
        help="The file path of h5ad or directory containing h5ad files. eg: /input/input.h5ad or /input/directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default=DEFAULT_OUTPUT_DIR,
        help="The output path. eg: /output",
    )
    parser.add_argument("--qc", action="store_true", help="Whether to perform quality control (QC)")
    parser.add_argument(
        "-gip", "--gene_info_path", default=GENE_INFO_PATH, type=Path, help="The gene info path."
    )
    parser.add_argument("-on", "--organ_name", type=str, default="example", help="Organ name.")
    parser.add_argument(
        "-n", "--num_proc", type=int, default=8, help="The num proc to process dataset."
    )
    parser.add_argument(
        "-hct",
        "--has_cell_type",
        action="store_true",
        help="Whether the dataset contains 'cell_type' column. If set, include it in metadata.",
    )

    return parser.parse_args()


class GeneInfoProcessor:
    """Handles loading and processing of gene information."""

    def __init__(self, gene_info_path: Path):
        """Initialize with path to gene info CSV file."""
        self.gene_info_path = gene_info_path
        self.gene_info = pd.read_csv(self.gene_info_path)

    def _convert_gene_info_to_dict(self) -> Dict[str, Dict]:
        return {
            "name_to_id": self.gene_info.set_index("gene_name")["ensembl_id"].to_dict(),
            "name_to_type": self.gene_info.set_index("gene_name")["gene_type"].to_dict(),
            "id_to_type": self.gene_info.set_index("ensembl_id")["gene_type"].to_dict(),
        }

    def get_functional_genes(self) -> List[str]:
        """Get list of functional genes (protein coding and miRNA)."""
        return self.gene_info[
            (self.gene_info["gene_type"] == "protein_coding")
            | (self.gene_info["gene_type"] == "miRNA")
        ]["ensembl_id"].tolist()


def ensure_directory_exists(path: Union[str, Path]) -> None:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    if not path.exists():
        logger.info(f"{path} does not exist, creating it")
        path.mkdir(parents=True, exist_ok=True)


def perform_qc(adata: AnnData, functional_genes: List[str], gene_name_id_map: Dict) -> AnnData:
    """Perform quality control on AnnData object."""
    # Set var_names appropriately
    if "feature_name" in adata.var:
        adata.var_names = adata.var["feature_name"].tolist()
    elif "gene_name" in adata.var:
        adata.var_names = adata.var["gene_name"].tolist()

    # Mitochondrial gene filtering
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    # Filter cells based on mitochondrial content
    mean_mt, std_mt = adata.obs["pct_counts_mt"].mean(), adata.obs["pct_counts_mt"].std()
    mt_range = (mean_mt - 3 * std_mt, mean_mt + 3 * std_mt)
    adata = adata[
        (adata.obs["pct_counts_mt"] >= mt_range[0]) & (adata.obs["pct_counts_mt"] <= mt_range[1]), :
    ]

    # Filter cells based on total counts
    mean_counts, std_counts = adata.obs["total_counts"].mean(), adata.obs["total_counts"].std()
    counts_range = (mean_counts - 3 * std_counts, mean_counts + 3 * std_counts)
    adata = adata[
        (adata.obs["total_counts"] >= counts_range[0])
        & (adata.obs["total_counts"] <= counts_range[1])
    ]

    # Doublet detection
    scrub = Scrublet(adata.X)
    _, predicted_doublets = scrub.scrub_doublets()
    if predicted_doublets is not None:
        adata = adata[~predicted_doublets]

    # Filter for functional genes
    functional_set = set(functional_genes)
    retain_genes = [
        gene
        for gene in adata.var_names
        if gene in functional_set or gene_name_id_map.get(gene, None) in functional_set
    ]

    adata = adata[:, retain_genes]
    sc.pp.filter_cells(adata, min_genes=7)

    return adata


def generate_task_id() -> str:
    """Generate a unique task ID with date and random string."""

    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M")
    random_str = "".join(
        random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", k=8)
    )
    return date_str + f"_{random_str}"


def preprocess_data(
    input_path: Path,
    output_path: Path,
    gene_info: Dict,
    gene_token_info: Dict,
    perform_qc_flag: bool = False,
) -> Path:
    """Preprocess single-cell data and save as loom file.

    Args:
        input_path: Path to either a single h5ad file or a directory containing h5ad files
        output_path: Output directory path
        gene_info: Dictionary containing gene information
        gene_token_info: Dictionary containing gene token information
        perform_qc_flag: Whether to perform quality control

    Returns:
        Path to the directory containing processed loom files
    """
    try:
        task_id = generate_task_id()
        task_dir = output_path / task_id
        ensure_directory_exists(task_dir)

        # Determine if input is a directory or single file
        if input_path.is_dir():
            h5ad_files = list(input_path.glob("*.h5ad"))
            if not h5ad_files:
                raise ValueError(f"No h5ad files found in directory: {input_path}")
        else:
            h5ad_files = [input_path]

        for h5ad_file in h5ad_files:
            file_name = h5ad_file.stem
            adata = sc.read_h5ad(h5ad_file)
            adata.var_names_make_unique()
            adata.obs_names_make_unique()

            logger.info(f"Processing file: {h5ad_file}, AnnData shape: {adata.shape}")

            if perform_qc_flag:
                adata = perform_qc(adata, gene_info["functional_genes"], gene_info["name_to_id"])

            # Process gene information
            gene_type_arr = []
            ensembl_id_arr = []
            pretrain_uncover_genes = []

            for gene_name in adata.var_names:
                if (
                    gene_name in gene_info["name_to_type"]
                    and gene_info["name_to_id"].get(gene_name, None) in gene_token_info
                ):
                    gene_type_arr.append(gene_info["name_to_type"][gene_name])
                    ensembl_id_arr.append(gene_info["name_to_id"][gene_name])
                elif str(gene_name).startswith("ENSG") and gene_name in gene_token_info:
                    gene_type_arr.append(gene_info["id_to_type"][gene_name])
                    ensembl_id_arr.append(gene_name)
                else:
                    pretrain_uncover_genes.append(gene_name)

            if pretrain_uncover_genes:
                logger.warning(f"Removing {len(pretrain_uncover_genes)} genes not in token file")
                adata = adata[:, adata.var_names.difference(pretrain_uncover_genes)]

            if adata.n_obs == 0:
                raise ValueError(
                    f"No cells remaining after filtering - check input data: {h5ad_file}"
                )

            # Save as loom file
            loom_path = task_dir / f"{file_name}.loom"
            loompy.create(
                str(loom_path),
                adata.X.T,
                {"ensembl_id": np.array(ensembl_id_arr), "gene_type": gene_type_arr},
                {
                    "n_counts": adata.X.sum(axis=1),
                    "barcode": np.array(adata.obs_names),
                    "cell_type": np.array(adata.obs["cell_type"]),
                },
            )

            logger.info(f"Successfully created loom file: {loom_path}")

        return task_dir

    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}\n{traceback.format_exc()}")
        raise


def tokenize_cell(gene_vector: np.ndarray, gene_tokens: np.ndarray) -> np.ndarray:
    """Convert normalized gene expression vector to tokenized rank value encoding."""

    nonzero_mask = np.nonzero(gene_vector)[0]
    sorted_indices = np.argsort(-gene_vector[nonzero_mask])
    return gene_tokens[nonzero_mask][sorted_indices]


class TranscriptomeTokenizer:
    """Handles tokenization of data."""

    def __init__(
        self,
        custom_attr_name_dict: Dict[str, str],
        nproc: int = 1,
        gene_median_file: Path = GENE_MEDIAN_FILE,
        token_dictionary_file: Path = TOKEN_DICTIONARY_FILE,
    ):
        """Initialize tokenizer with necessary dictionaries and parameters."""
        self.custom_attr_name_dict = custom_attr_name_dict
        self.nproc = nproc

        # Load gene normalization factors
        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)

        # Load token dictionary
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        # Create gene list dictionary for filtering
        self.gene_keys = list(self.gene_median_dict.keys())
        self.genelist_dict = dict.fromkeys(self.gene_keys, True)

    def tokenize_data(
        self, loom_data_directory: Path, output_directory: Path, output_prefix: str
    ) -> None:
        """Tokenize loom files and save as tokenized dataset."""
        tokenized_cells, cell_metadata = self._process_loom_files(loom_data_directory)
        tokenized_dataset = self._create_dataset(tokenized_cells, cell_metadata)

        output_path = output_directory / f"{output_prefix}.dataset"
        tokenized_dataset.save_to_disk(str(output_path), max_shard_size="5000MB")

    def _process_loom_files(self, loom_dir: Path) -> Tuple[List, Dict]:
        """Process all loom files in directory."""
        tokenized_cells = []
        cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}

        for loom_file in loom_dir.glob("*.loom"):
            file_tokens, file_metadata = self._process_single_loom(loom_file)
            tokenized_cells.extend(file_tokens)

            for k, v in self.custom_attr_name_dict.items():
                cell_metadata[v].extend(file_metadata[k])

        return tokenized_cells, cell_metadata

    def _process_single_loom(self, loom_path: Path) -> Tuple[List, Dict]:
        """Process a single loom file."""
        file_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.keys()}
        tokenized_cells = []

        with loompy.connect(str(loom_path)) as data:
            # Get protein-coding and miRNA genes
            coding_miRNA_loc = np.array(
                [self.genelist_dict.get(i, False) for i in data.ra["ensembl_id"]]
            )
            coding_miRNA_loc = np.where(coding_miRNA_loc)[0]

            norm_factors = np.array(
                [self.gene_median_dict[i] for i in data.ra["ensembl_id"][coding_miRNA_loc]]
            )
            gene_tokens = np.array(
                [self.gene_token_dict[i] for i in data.ra["ensembl_id"][coding_miRNA_loc]]
            )

            # Get cells passing filters
            try:
                filter_pass = np.array(data.ca["filter_pass"], dtype=bool)
            except AttributeError:
                logger.warning(f"{loom_path} has no 'filter_pass' attribute, processing all cells")
                filter_pass = np.ones(data.shape[1], dtype=bool)

            # Process cells in batches
            for _, _, view in data.scan(items=np.where(filter_pass)[0], axis=1):
                subview = view.view[coding_miRNA_loc, :]
                normalized = subview[:, :] / subview.ca.n_counts * 10_000 / norm_factors[:, None]

                tokenized_cells.extend(
                    [
                        tokenize_cell(normalized[:, i], gene_tokens)
                        for i in range(normalized.shape[1])
                    ]
                )

                for k in file_metadata:
                    file_metadata[k].extend(subview.ca[k].tolist())

        return tokenized_cells, file_metadata

    def _create_dataset(self, tokenized_cells: List, cell_metadata: Dict) -> Dataset:
        """Create and process the final dataset."""
        dataset_dict = {"input_ids": tokenized_cells}
        dataset_dict.update(cell_metadata)

        dataset = Dataset.from_dict(dataset_dict)

        def truncate(example):
            example["input_ids"] = example["input_ids"][:2048]
            return example

        truncated_dataset = dataset.map(truncate, num_proc=self.nproc)

        def add_length(example):
            example["length"] = len(example["input_ids"])
            return example

        return truncated_dataset.map(add_length, num_proc=self.nproc)


def main(args: argparse.Namespace) -> None:
    """Main processing pipeline."""
    try:
        # Initialize and load gene information
        gene_processor = GeneInfoProcessor(args.gene_info_path)
        gene_info = {
            **gene_processor._convert_gene_info_to_dict(),
            "functional_genes": gene_processor.get_functional_genes(),
        }

        # Load token information
        with open(TOKEN_DICTIONARY_FILE, "rb") as f:
            gene_token_info = pickle.load(f)

        # Preprocess data
        ensure_directory_exists(args.output)
        input_path = Path(args.input)

        # Check if input exists
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        task_dir = preprocess_data(
            input_path, Path(args.output), gene_info, gene_token_info, args.qc
        )

        # Tokenize data
        logger.info("Starting tokenization")

        # Set custom_attr_name_dict based on --has_cell_type flag
        if args.has_cell_type:
            custom_attr_name_dict = {"barcode": "barcode", "cell_type": "cell_type"}
        else:
            custom_attr_name_dict = {"barcode": "barcode"}

        tokenizer = TranscriptomeTokenizer(
            custom_attr_name_dict=custom_attr_name_dict, nproc=args.num_proc
        )
        tokenizer.tokenize_data(task_dir, task_dir, args.organ_name)

        logger.info(f"Processing complete. Results in: {task_dir}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
