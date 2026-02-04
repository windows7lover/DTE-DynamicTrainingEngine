import warnings
import torch
import torch.utils._pytree as pytree
from dataclasses import is_dataclass, fields


def try_register_tensor_dataclass_as_pytree(cls) -> bool:
    """
    Try to register a dataclass as a PyTree.

    Conditions:
      - must be a dataclass
      - all fields must be torch.Tensor

    Returns:
      True  -> successfully registered
      False -> not registered (warning emitted)
    """

    if not is_dataclass(cls):
        warnings.warn(
            f"[torch.compile] {cls.__name__} is not a dataclass; "
            "cannot be registered as a PyTree. "
            "Compiled paths will be disabled for this memory type."
        )
        return False

    tensor_fields = []
    for f in fields(cls):
        tensor_fields.append(f.name)

    def flatten(obj):
        children = []
        for name in tensor_fields:
            val = getattr(obj, name)
            if not isinstance(val, torch.Tensor):
                raise TypeError(
                    f"[torch.compile] Field '{name}' of {cls.__name__} "
                    f"is not a torch.Tensor (got {type(val)})."
                )
            children.append(val)
        return tuple(children), None

    def unflatten(aux_data, children):
        return cls(*children)

    try:
        pytree.register_pytree_node(cls, flatten, unflatten)
    except Exception as e:
        warnings.warn(
            f"[torch.compile] Failed to register {cls.__name__} as PyTree: {e}. "
            "Compiled paths will be disabled for this memory type."
        )
        return False

    return True
