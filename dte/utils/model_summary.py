import torch
import torch.nn as nn


BANNER = "#" * 100


def _collect_params(module: nn.Module, prefix: str):
    """
    Collect (name, tensor) pairs for all parameters under a module.
    """
    out = []
    for name, p in module.named_parameters(recurse=True):
        out.append((f"{prefix}.{name}", p))
    return out


def print_param_table(model):
    """
    Print parameter tables
    """
    if hasattr(model, "module"):
        model = model.module

    def print_section(title, rows):
        print(title)
        print("NAME".ljust(60), "SHAPE".ljust(25), "PARAMS")
        print("-" * 90)
        for name, p in rows:
            shape = tuple(p.shape)
            num = p.numel()
            print(f"{name.ljust(60)} {str(shape).ljust(25)} {num}")
        print() 

    # -------------------------------
    # Gather sections
    # -------------------------------
    enc_rows = _collect_params(model.encoder, "_orig_mod.encoder")
    mem_rows = _collect_params(model.memory_init, "_orig_mod.memory_init")
    core_rows = _collect_params(model.core, "_orig_mod.core")

    # -------------------------------
    # Print with banners
    # -------------------------------
    print("\n" * 2)
    print(BANNER)
    print("MODEL PARAMETER SUMMARY".center(len(BANNER)))
    print(BANNER)
    print()

    print_section("ENCODER PARAMETERS", enc_rows)
    print_section("MEMORY INITIALIZER PARAMETERS", mem_rows)
    print_section("CORE PARAMETERS", core_rows)

    total = sum(p.numel() for _, p in enc_rows + mem_rows + core_rows)

    print("TOTAL PARAMS:", total)
    print()
    print(BANNER)
    print("\n" * 2)
