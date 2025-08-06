"""
ScDifformer script for cell annotation tasks.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict

import dill
import pandas as pd
from datasets import load_from_disk
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, set_seed

from libs import DataCollatorForCellClassification, DEFAULT_OUTPUT_DIR

current = datetime.now().strftime("%Y%m%d_%H%M")
PREDICT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / f"{current}_predict"


def save_excel(ds, name_id: Dict[str, int], preds, out_path: Path, has_labels: bool = True) -> None:
    """Save predictions to an Excel file. If no labels, omit the 'Cell_type' column."""
    id_name = {v: k for k, v in name_id.items()}

    data = {"Barcode": list(ds["barcode"]), "Pred_cell_type": [id_name[i] for i in preds]}
    if has_labels:
        data["Cell_type"] = [id_name[i] for i in ds["label"]]  # Add true labels only if available

    df = pd.DataFrame(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)


def compute_metrics(eval_pred):
    """Compute evaluation metrics. Skip if no labels are provided."""
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    if labels is None or (isinstance(labels, list) and not labels):  # No labels available
        logger.warning("⚠️ No labels provided, skipping metrics computation.")
        return {}  # Return empty dict, no metrics computed

    # Compute accuracy and macro F1
    return dict(
        accuracy=accuracy_score(labels, preds),
        macro_f1=f1_score(labels, preds, average="macro"),
    )


def predict(args):
    set_seed(args.random_seed)

    test_ds_path = args.test_ds_path
    logger.info(f"🔍 Loading test set: {test_ds_path}")
    testset = load_from_disk(str(test_ds_path))

    # Check if dataset has labels (either 'cell_type' or 'label' column)
    has_labels = "cell_type" in testset.column_names or "label" in testset.column_names

    name_id_path = Path(args.name_id_path)
    logger.info(f"🔍 Loading name↔id dict: {name_id_path}")
    name_id: Dict[str, int] = dill.load(open(name_id_path, "rb"))

    if has_labels:
        # Rename 'cell_type' to 'label' if necessary
        if "cell_type" in testset.column_names:
            testset = testset.rename_column("cell_type", "label")
        # Map label names to IDs
        testset = testset.map(lambda x: {"label": name_id[x["label"]]}, num_proc=args.num_proc)
        logger.info("✅ Dataset has labels, mapped to IDs.")
    else:
        # No labels: Add a dummy 'label' column (-100 to be ignored by Trainer)
        testset = testset.map(lambda x: {"label": -100}, num_proc=args.num_proc)
        logger.warning(
            "⚠️ Dataset has no 'cell_type' or 'label'. Running in prediction-only mode. Metrics will be skipped."
        )

    # ---------- Load model ----------
    model_dir = Path(args.fine_tune_model_path)
    logger.info(f"🔍 Loading model: {model_dir}")
    model = (
        BertForSequenceClassification.from_pretrained(str(model_dir), num_labels=len(name_id))
        .to("cuda")
        .eval()
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(model_dir),
            per_device_eval_batch_size=args.batch_size,
            dataloader_num_workers=args.num_proc,
            do_train=False,
            do_eval=False,
        ),
        data_collator=DataCollatorForCellClassification(),
        compute_metrics=compute_metrics,
    )

    logger.info("🚀 Predicting ...")
    preds = trainer.predict(testset)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw predictions
    pickle_path = out_dir / f"predictions_{args.organ_name}.pkl"
    with open(pickle_path, "wb") as fp:
        pickle.dump(preds, fp)
    logger.info(f"✅ raw preds -> {pickle_path}")

    # Save Excel (compatible with no labels)
    excel_path = out_dir / f"pred_{args.organ_name}.xlsx"
    save_excel(testset, name_id, preds.predictions.argmax(-1), excel_path, has_labels=has_labels)
    logger.info(f"✅ excel     -> {excel_path}")

    # Save metrics only if labels are available and metrics were computed
    if has_labels and preds.metrics:
        metrics_path = out_dir / f"metrics_{args.organ_name}.json"
        with open(metrics_path, "w") as f:
            json.dump(preds.metrics, f, indent=2)
        logger.info(f"📊 metrics   -> {preds.metrics}")
    else:
        logger.info("📊 No metrics computed (no labels available).")


# ----------------- CLI ----------------- #
def parse_args():
    ap = argparse.ArgumentParser("Cell-type prediction")
    ap.add_argument("-on", "--organ_name", type=str, default="example")
    ap.add_argument("-tdp", "--test_ds_path", type=str)
    ap.add_argument("-nip", "--name_id_path", type=str)
    ap.add_argument("-ftmp", "--fine_tune_model_path", type=str)
    ap.add_argument("-o", "--output_dir", type=str, default=PREDICT_OUTPUT_DIR)
    ap.add_argument("-bs", "--batch_size", type=int, default=4)
    ap.add_argument("-np", "--num_proc", type=int, default=8)
    ap.add_argument("-rs", "--random_seed", type=int, default=2025)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args)
