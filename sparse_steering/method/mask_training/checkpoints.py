"""Save mask checkpoints atomically and exclude zero direction coefficients."""

from __future__ import annotations

from pathlib import Path

import torch


def atomic_torch(path, value):
    temp = Path(str(path) + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def rankable_mask_logits(logits, dense, k):
    if int((dense != 0).sum()) < k:
        raise ValueError(
            "Dense direction has too few nonzero coordinates for the required exact-k mask."
        )
    return logits.masked_fill(dense == 0, -1000000000.0)
