# ScDifformer
A single cell large language model with context-awareness, enhanced by a denoising diffusion module and a dedicated post-training phase.

## Quick Start

1. Clone the code

```shell
git clone https://github.com/yaoqiqi17732/scDifformer.git
cd scDifformer
```

2. Install dependencies
   Create environment and install dependencies via [uv](https://docs.astral.sh/uv)

```shell
uv sync
```

3. Example usage

   Note: Demonstration examples only - replace with your content and customize per instructions.
    1. Preprocess
    ```python
    uv run python preprocess/preprocess.py -hct
    ```
    2. Fine-tune

    ```python
    uv run python fine_tune/fine_tune_celltype_annotation.py \
        -pmd /path/to/pretrained_model \
        -id /path/to/arrow_data \
        -nip /path/to/name_id.pkl
    ```
    3. Inference
    ```python
    uv run python inference/inference_celltype_annotation.py \
        -tdp /data/test_dataset.h5ad \
        -nip /path/to/name_id.pkl \
        -ftmp /path/to/fine_tune_model
    ```

## License

ScDifformer is open-sourced under the [MIT License](https://opensource.org/licenses/MIT).
