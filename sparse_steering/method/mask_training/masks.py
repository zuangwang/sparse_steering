"""Build exact-k masks and propagate gradients through a sigmoid surrogate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np
import torch


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def exact_topk_mask(scores: Sequence[float] | np.ndarray, k: int) -> np.ndarray:
    """Select exactly k coordinates, breaking equal-score ties by lower index."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Scores must be a non-empty one-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError("Scores must contain only finite values.")
    count = _validate_k(k, int(values.size))
    indices = np.arange(values.size, dtype=np.int64)
    order = np.lexsort((indices, -values))
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    mask.flags.writeable = False
    return mask


def _finite_vector(value: torch.Tensor | Sequence[float], name: str) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional tensor")
    if tensor.dtype is torch.bool or not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float64)
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _validate_k(k: object, dimension: int) -> int:
    if not isinstance(k, Integral) or isinstance(k, bool):
        raise ValueError("K must be an integer.")
    count = int(k)
    if not 1 <= count <= dimension:
        raise ValueError(f"k must lie in [1, {dimension}]")
    return count


def stable_topk_mask(scores: torch.Tensor | Sequence[float], k: int) -> torch.Tensor:
    values = _finite_vector(scores, "scores")
    count = _validate_k(k, int(values.numel()))
    detached = values.detach().to(device="cpu", dtype=torch.float64).numpy()
    numpy_mask = exact_topk_mask(detached, count)
    return torch.as_tensor(np.array(numpy_mask, copy=True), dtype=torch.bool, device=values.device)


@dataclass(frozen=True)
class StraightThroughExactKDirection:
    direction: torch.Tensor
    hard_mask: torch.Tensor
    gate: torch.Tensor
    soft_probabilities: torch.Tensor


def straight_through_exact_k_direction(
    mask_logits: torch.Tensor,
    dense_direction: torch.Tensor | Sequence[float],
    *,
    k: int,
    temperature: float = 1.0,
    unit_tolerance: float = 1e-05,
) -> StraightThroughExactKDirection:
    """Use the binary top-k mask in the forward pass and a sigmoid surrogate for its gradients."""
    if not torch.is_tensor(mask_logits) or not mask_logits.is_floating_point():
        raise ValueError("Mask logits must be a floating-point Torch tensor.")
    logits = _finite_vector(mask_logits, "mask_logits")
    dense = _finite_vector(dense_direction, "dense_direction").to(
        device=logits.device, dtype=logits.dtype
    )
    if dense.shape != logits.shape:
        raise ValueError("Mask logits and the dense direction must have the same shape.")
    count = _validate_k(k, int(logits.numel()))
    if (
        not isinstance(temperature, Real)
        or isinstance(temperature, bool)
        or (not math.isfinite(float(temperature)))
        or (float(temperature) <= 0.0)
    ):
        raise ValueError("Temperature must be finite and positive.")
    if (
        not isinstance(unit_tolerance, Real)
        or isinstance(unit_tolerance, bool)
        or (not math.isfinite(float(unit_tolerance)))
        or (float(unit_tolerance) < 0.0)
    ):
        raise ValueError("The unit-norm tolerance must be finite and nonnegative.")
    dense_norm = float(torch.linalg.vector_norm(dense.detach()).item())
    if not math.isclose(dense_norm, 1.0, rel_tol=0.0, abs_tol=float(unit_tolerance)):
        raise ValueError(
            f"dense_direction must have unit Euclidean norm; observed {dense_norm:.17g}"
        )
    soft = torch.sigmoid(logits / float(temperature))
    hard = stable_topk_mask(logits, count)
    if bool(torch.any(dense.detach()[hard] == 0).item()):
        raise ValueError("Selected support contains zero dense-direction coordinates.")
    hard_values = hard.to(dtype=logits.dtype)
    # The forward value is hard; autograd sees only the sigmoid derivative.
    gate = hard_values + soft - soft.detach()
    masked = gate * dense
    norm = torch.linalg.vector_norm(masked)
    forward_norm = float(norm.detach().item())
    if not math.isfinite(forward_norm) or forward_norm <= 0.0:
        raise ValueError("Hard-masked dense direction has zero or non-finite norm.")
    direction = masked / norm
    return StraightThroughExactKDirection(
        direction=direction, hard_mask=hard, gate=gate, soft_probabilities=soft
    )
