# datasets/base_dataset_metadata.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class BaseDatasetMetadata:
    """
    Base dataset_metadata class.
    All datasets define their own subclass.
    """

    @classmethod
    def from_config(cls, cfg: Any) -> "BaseDatasetMetadata":
        """
        Construct dataset_metadata from a config node (OmegaConf or dict-like).
        Assumes cfg contains fields matching the subclass __init__ signature.
        """
        if hasattr(cfg, "items"):
            data = dict(cfg)
        else:
            data = dict(cfg)

        return cls(**data)
