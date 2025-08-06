from datetime import datetime
from pathlib import Path

GENE_MEDIAN_FILE = Path(__file__).parent.parent / "data" / "gene_median_dictionary.pkl"
TOKEN_DICTIONARY_FILE = Path(__file__).parent.parent / "data" / "token_dictionary.pkl"
GENE_INFO_PATH = Path(__file__).parent.parent / "data" / "gene_info_table.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
DEFAULT_MODEL_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "model"
EXAMPLE_DATA_DIR = Path(__file__).parent.parent / "data"
EXAMPLE_DATASET_DIR = (
    Path(__file__).parent.parent / "data" / "output" / "20250806_0949_duUWNQUz" / "example.dataset"
)
EXAMPLE_NAME_ID_PATH = Path(__file__).parent.parent / "data" / "name_id.pkl"

current = datetime.now().strftime("%Y%m%d_%H%M")
CURRENT_MODEL_OUTPUT_DIR = DEFAULT_MODEL_OUTPUT_DIR / current
