"""Convert a strength setting to an edit norm and realize a sparse additive edit."""

from __future__ import annotations

import math

import torch


def edit_norm(edit_norm_ratio_squared, scale):
    if (
        not math.isfinite(edit_norm_ratio_squared)
        or edit_norm_ratio_squared < 0
        or (not math.isfinite(scale))
        or (scale <= 0)
    ):
        raise ValueError(
            "Squared edit norm ratio must be nonnegative and scale positive, both finite."
        )
    return math.sqrt(edit_norm_ratio_squared) * scale


@torch.no_grad()
def realize_batch(states, direction, target_norm):
    """Match the sparse additive displacement to the requested norm in bf16.

    Search the additive multiplier using the displacement after bf16 rounding.
    The later coordinate adjustments improve the match without changing support.
    """
    if states.ndim != 2 or states.dtype != torch.bfloat16:
        raise ValueError("States must be a BF16 matrix.")
    if direction.ndim != 1 or direction.numel() != states.shape[1]:
        raise ValueError("Direction must match hidden dimension.")
    if not math.isfinite(target_norm) or target_norm <= 0:
        raise ValueError("Target norm must be finite and positive.")
    direction = direction.float()
    norm = direction.norm()
    if not bool(torch.isfinite(states).all() & torch.isfinite(direction).all()) or norm <= 1e-10:
        raise ValueError("Invalid states or direction.")
    direction = direction / norm
    target = target_norm**2
    base = states.float()

    def candidate(scales):
        # Paper sparse-edit equation: h_tilde = h + alpha * v_M.
        delta = scales[:, None] * target_norm * direction[None, :]
        # Casting the increment and adding it each incur bf16 rounding.
        updated = states + delta.to(torch.bfloat16)
        squared_edit_norm = (updated.float() - base).square().sum(-1)
        return updated, squared_edit_norm

    count = len(states)
    one = torch.ones(count, device=states.device)
    best, squared_edit_norm = candidate(one)
    best_error = (squared_edit_norm - target).abs()
    low = torch.zeros_like(one)
    high = one.clone()
    high_squared_edit_norm = squared_edit_norm
    for _ in range(32):
        needs = high_squared_edit_norm < target
        if not bool(needs.any()):
            break
        high = torch.where(needs, high * 2, high)
        _, high_squared_edit_norm = candidate(high)
    else:
        raise RuntimeError("Failed to bracket BF16 squared edit norm.")
    for _ in range(20):
        mid = (low + high) / 2
        updated, mid_squared_edit_norm = candidate(mid)
        error = (mid_squared_edit_norm - target).abs()
        better = error < best_error
        best = torch.where(better[:, None], updated, best)
        best_error = torch.minimum(best_error, error)
        low = torch.where(mid_squared_edit_norm < target, mid, low)
        high = torch.where(mid_squared_edit_norm >= target, mid, high)
    realized = best.float() - base
    squared_edit_norm = realized.square().sum(-1)
    cosine = (realized * direction).sum(-1) / squared_edit_norm.sqrt().clamp_min(1e-12)
    return best, squared_edit_norm, (squared_edit_norm - target).abs() / target, cosine


@torch.no_grad()
def realize_quantized_addition(states, direction, target_norm, tolerance=0.0005, max_steps=64):
    updated, squared_edit_norm, error, _ = realize_batch(states, direction, target_norm)
    direction = direction.float()
    direction = direction / direction.norm()
    base = states.float()
    target = target_norm**2
    rounding_adjustments = torch.zeros(len(states), dtype=torch.int32, device=states.device)
    away = torch.where(direction > 0, torch.inf, -torch.inf).to(states.dtype)[None, :]
    support = (direction != 0)[None, :]
    # Move one supported coordinate by one representable value only when it
    # improves the norm match and preserves its displacement sign.
    for _ in range(max_steps):
        needed = error > tolerance
        if not bool(needed.any()):
            break
        displacement = updated.float() - base
        toward = torch.where((squared_edit_norm < target)[:, None], away, -away)
        adjacent = torch.nextafter(updated, toward)
        next_displacement = adjacent.float() - base
        candidate_squared_edit_norm = (
            squared_edit_norm[:, None] + next_displacement.square() - displacement.square()
        )
        valid = support & (next_displacement * direction >= 0) & torch.isfinite(adjacent)
        candidate_error = torch.where(
            valid, (candidate_squared_edit_norm - target).abs(), torch.inf
        )
        best_error, coordinate = candidate_error.min(-1)
        improve = needed & (best_error < (squared_edit_norm - target).abs())
        if not bool(improve.any()):
            break
        rows = torch.nonzero(improve, as_tuple=True)[0]
        updated[rows, coordinate[rows]] = adjacent[rows, coordinate[rows]]
        rounding_adjustments += improve.to(torch.int32)
        squared_edit_norm = (updated.float() - base).square().sum(-1)
        error = (squared_edit_norm - target).abs() / target
    displacement = updated.float() - base
    cosine = (displacement * direction).sum(-1) / squared_edit_norm.sqrt().clamp_min(1e-12)
    return updated, squared_edit_norm, error, cosine, rounding_adjustments
