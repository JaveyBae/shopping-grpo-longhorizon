#!/usr/bin/env python3
"""Patch pinned veRL to use its torch padding fallback on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


ORIGINAL_SHA256 = "8da49fde853e5b5748db887bf2602fc290bb2a514041c9589482bb0d846d2ec0"
BACKUP_SUFFIX = ".shopping-grpo-attention.orig"
MARKER = "SHOPPING_GRPO_TORCH_PADDING_FALLBACK_V1"
OLD = "from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input"
NEW = (
    "# SHOPPING_GRPO_TORCH_PADDING_FALLBACK_V1: FlashAttention is optional.\n"
    "        from verl.utils.npu_flash_attn_utils import (\n"
    "            index_first_axis,\n"
    "            pad_input,\n"
    "            rearrange,\n"
    "            unpad_input,\n"
    "        )"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_target() -> Path:
    import verl

    return Path(verl.__file__).resolve().parent / "utils" / "attention_utils.py"


def apply_patch(target: Path) -> None:
    source = target.read_text(encoding="utf-8")
    if MARKER in source:
        print(f"veRL attention fallback patch already applied: {target}")
        return
    if sha256(target) != ORIGINAL_SHA256 or OLD not in source:
        raise RuntimeError(f"refusing to patch unknown attention_utils.py: {target}")

    backup = Path(f"{target}{BACKUP_SUFFIX}")
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"applied veRL attention fallback patch: {target}")
    print(f"backup: {backup}")
    print(f"patched_sha256: {sha256(target)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path)
    args = parser.parse_args()
    apply_patch(args.target or locate_target())


if __name__ == "__main__":
    main()
