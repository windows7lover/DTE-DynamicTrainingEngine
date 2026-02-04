# build_and_pack.py
import os
import shutil
import tempfile
import numpy as np
import importlib

from build_sudoku_dataset import DataProcessConfig


def import_builder(path: str):
    """
    Import a builder function like 'module:object'.
    Handles Argdantic @cli.command decorators by unwrapping .callback.
    """
    if ":" in path:
        module_name, obj_name = path.split(":")
    else:
        module_name, obj_name = path.rsplit(".", 1)

    module = importlib.import_module(module_name)
    obj = getattr(module, obj_name)

    # Argdantic Command object → unwrap actual Python function
    if hasattr(obj, "callback"):
        return obj.callback

    return obj


def pack_split(folder):
    """
    Convert HF-style raw split (all__inputs.npy, all__labels.npy)
    into TRM-compatible dataset.npy with keys {"inputs", "labels"}.
    """
    inputs_path = os.path.join(folder, "all__inputs.npy")
    labels_path = os.path.join(folder, "all__labels.npy")

    if not os.path.exists(inputs_path) or not os.path.exists(labels_path):
        print(f"[WARN] Missing 'all__inputs.npy' or 'all__labels.npy' in {folder}, skipping.")
        return

    inputs = np.load(inputs_path)
    labels = np.load(labels_path)

    arr = {"inputs": inputs, "labels": labels}
    out_path = os.path.join(folder, "dataset.npy")
    np.save(out_path, arr)

    print(f"[PACKED] {out_path}")


def build_and_pack(
    builder_path: str,
    builder_kwargs: dict,
    output_dataset_dir: str,
):
    """
    1. Build dataset using builder(config)
    2. Convert raw HF-style splits to TRM dataset.npy
    3. Copy final dataset folder to output_dataset_dir
    """

    # --------------------------------------------------------
    # Step 1: Build in temporary directory
    # --------------------------------------------------------
    tmp = tempfile.mkdtemp(prefix="dataset_build_")
    print(f"[TEMP DIR] {tmp}")

    # Import builder function (unwrap if needed)
    builder = import_builder(builder_path)

    # Convert dict → DataProcessConfig
    cfg_dict = dict(builder_kwargs["config"])
    cfg = DataProcessConfig(**cfg_dict)

    # Inject output directory into config
    cfg.output_dir = tmp

    print("[BUILD] Running dataset builder...")
    builder(config=cfg)

    # --------------------------------------------------------
    # Step 2: Pack splits
    # --------------------------------------------------------
    print("[PACK] Converting splits to TRM dataset format...")

    for split in ["train", "test"]:
        split_path = os.path.join(tmp, split)
        if os.path.isdir(split_path):
            pack_split(split_path)
        else:
            print(f"[WARN] Split '{split}' not found at {split_path}")

    # --------------------------------------------------------
    # Step 3: Copy built dataset to destination
    # --------------------------------------------------------
    print(f"[COPY] Copying final dataset to: {output_dataset_dir}")

    if os.path.exists(output_dataset_dir):
        shutil.rmtree(output_dataset_dir)

    shutil.copytree(tmp, output_dataset_dir)

    print("[DONE] Dataset built and packed successfully.")
    print(f"[RESULT] {output_dataset_dir}")


# ------------------------------------------------------------
# Execute with your exact arguments
# ------------------------------------------------------------
build_and_pack(
    builder_path="build_sudoku_dataset:preprocess_data",
    builder_kwargs={
        "config": {
            "source_repo": "sapientinc/sudoku-extreme",
            "subsample_size": 1000,
            "min_difficulty": None,
            "num_aug": 1000,
        }
    },
    output_dataset_dir="build_dataset/sudoku-extreme-1k-aug-1000"
)
