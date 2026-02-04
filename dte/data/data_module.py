# data_module.py
from __future__ import annotations

from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dte.utils.config_loader import import_from_path
from dte.data.batch import Batch
from dte.data.batch_stream import BatchStream
from dte.data.fixed_numpy_dataset import FixedNumpyDataset
from dte.data.online_dataset import OnlineDataset
from dte.data.base_dataset_metadata import BaseDatasetMetadata


class DataModule:
    """
    Unified data module for online + fixed datasets.

    ONLINE MODE:
        - OnlineDataset internally generates a *batch* of size `batch_size`.
        - DataLoader is constructed with batch_size=None, so each iteration
          yields exactly one Batch (inputs [B, ...], labels [B, ...]).
        - A BatchStream is created for train and eval to provide exactly-n
          sample streaming on demand.

    FIXED MODE:
        - Standard PyTorch dataset / DataLoader with batch_size = `batch_size`,
          using a custom collate_fn to produce a Batch.
        - BatchStream works identically by slicing from cached loader batches.

    Device semantics:
        - BatchStream returns tensors on `self.device`.
        - In DDP, instantiate one DataModule per rank and pass the rank-local device.
    """

    def __init__(
        self,
        dataset_config,  # DatasetConfig
        batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
        device: torch.device,
    ):
        self.dataset_config = dataset_config
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.mode = dataset_config.mode

        self.device = device

        self.train_loader: DataLoader | None = None
        self.eval_loader: DataLoader | None = None

        # Exposed streams (created in setup())
        self.train_stream: BatchStream | None = None
        self.eval_stream: BatchStream | None = None

        # Build dataset_metadata from YAML class path
        dataset_metadata_cls = import_from_path(dataset_config.dataset_metadata_cls_path)
        assert issubclass(dataset_metadata_cls, BaseDatasetMetadata), (
            f"{dataset_metadata_cls} must inherit from BaseDatasetMetadata"
        )
        self.dataset_metadata = dataset_metadata_cls.from_config(
            dataset_config.dataset_metadata
        )

        self.train_size = None
        self.eval_size = None

    # ============================================================
    # ONLINE MODE  (batch-generating OnlineDataset)
    # ============================================================
    def _build_online(self):
        online_cfg = self.dataset_config.online
        assert online_cfg is not None

        # 1. Import generator class dynamically
        gen_cls = import_from_path(online_cfg.generator_cls_path)
        assert callable(gen_cls), f"Invalid generator class: {gen_cls}"

        # 2. Extract parameters
        gen_params: Dict = dict(online_cfg.generator)
        gen_params.setdefault("seed", self.seed)

        # 3. Instantiate generator
        generator = gen_cls(dataset_metadata=self.dataset_metadata, **gen_params)

        # Base seed folds in rank to ensure different streams per rank
        base_seed = int(self.seed + self.rank * 10_000_000)

        train_ds = OnlineDataset(
            generator_fn=generator,
            dataset_metadata=self.dataset_metadata,
            internal_batch_size=self.batch_size,
            base_seed=base_seed,
        )
        eval_ds = OnlineDataset(
            generator_fn=generator,
            dataset_metadata=self.dataset_metadata,
            internal_batch_size=self.batch_size,
            base_seed=base_seed + 1,
        )

        loader_kwargs = dict(
            batch_size=None,  # dataset already returns [B, ...]
            num_workers=4,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=4,
        )

        self.train_loader = DataLoader(train_ds, **loader_kwargs)
        self.eval_loader = DataLoader(eval_ds, **loader_kwargs)

        # infinite / nominal
        self.train_size = None
        self.eval_size = None

    # ============================================================
    # FIXED MODE  (standard, with DistributedSampler)
    # ============================================================
    def _build_fixed(self):
        fixed_cfg = self.dataset_config.fixed
        assert fixed_cfg is not None, "Fixed mode requires fixed.* config."

        train_ds = FixedNumpyDataset(fixed_cfg.train_path, self.dataset_metadata)
        eval_ds = FixedNumpyDataset(fixed_cfg.eval_path, self.dataset_metadata)

        def collate_batches(batch_list):
            inputs = torch.stack([b.inputs for b in batch_list], dim=0)
            labels = torch.stack([b.labels for b in batch_list], dim=0)
            if batch_list[0].sample_id is not None:
                sample_id = torch.stack([b.sample_id for b in batch_list], dim=0).view(-1)
            else:
                sample_id = None
            return Batch(inputs=inputs, labels=labels, sample_id=sample_id)

        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.seed,
        )
        eval_sampler = DistributedSampler(
            eval_ds,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )

        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=collate_batches,
        )

        self.train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kwargs)
        self.eval_loader = DataLoader(eval_ds, sampler=eval_sampler, **loader_kwargs)

        self.train_size = len(train_ds) // self.world_size
        self.eval_size = len(eval_ds) // self.world_size

    # ============================================================
    def setup(self):
        if self.mode == "online":
            self._build_online()
        elif self.mode == "fixed":
            self._build_fixed()
        else:
            raise ValueError(f"Unknown dataset mode '{self.mode}'")

        assert self.train_loader is not None
        assert self.eval_loader is not None

        self.train_stream = BatchStream(self.train_loader, self.device)
        self.eval_stream = BatchStream(self.eval_loader, self.device)
        self.train_stream.reset()
        self.eval_stream.reset()

    # ============================================================
    # Epoch hook (for DistributedSampler)
    # ============================================================
    def on_epoch_start(self, epoch: int):
        """
        For fixed datasets with DistributedSampler: reshuffle per epoch and
        reset train stream.

        Online datasets: no-op.
        """
        if self.mode != "fixed":
            return

        assert self.train_loader is not None
        sampler = getattr(self.train_loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        # Reset the train stream so it uses the reshuffled sampler order.
        if self.train_stream is not None:
            self.train_stream.reset()
