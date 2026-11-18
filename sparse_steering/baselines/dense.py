"""Match the requested additive displacement after bf16 rounding."""

from __future__ import annotations

import math

import torch

from sparse_steering.method.sparse_edit import realize_quantized_addition

QUANTIZATION_DIAGNOSTICS = []


@torch.no_grad()
def _realize_dense(states, direction, squared_edit_norm_limit):
    if not math.isfinite(squared_edit_norm_limit) or squared_edit_norm_limit <= 0:
        raise ValueError("Positive finite squared edit norm limit required.")
    y, _, _, _, repair = realize_quantized_addition(
        states, direction, math.sqrt(0.995 * squared_edit_norm_limit)
    )
    delta = y.double() - states.double()
    v = direction.double() / direction.double().norm()
    squared_edit_norm = delta.square().sum(-1)
    cosine = (delta * v).sum(-1) / squared_edit_norm.sqrt().clamp_min(1e-300)
    error = abs(squared_edit_norm / squared_edit_norm_limit - 1)
    before, after = (states.double().norm(dim=-1), y.double().norm(dim=-1))
    if (
        not bool(torch.isfinite(y).all())
        or bool((squared_edit_norm > squared_edit_norm_limit).any())
        or bool((error > 0.01).any())
        or bool((cosine < 0.9).any())
    ):
        raise ValueError("Dense realized squared edit norm and cosine audit failed.")
    return y, dict(
        realized_squared_edit_norm=squared_edit_norm,
        requested_squared_edit_norm=torch.full_like(squared_edit_norm, squared_edit_norm_limit),
        squared_edit_norm_error=error,
        cosine=cosine,
        norm_before=before,
        norm_after=after,
        effective_projection=(delta * v).sum(-1),
        quantization_adjustment_steps=repair.double(),
    )


def geometry(states, updated, direction, squared_edit_norm_limit):
    delta = updated.double() - states.double()
    unit = direction.double() / direction.double().norm()
    squared_edit_norm = delta.square().sum(-1)
    cosine = (delta * unit).sum(-1) / squared_edit_norm.sqrt().clamp_min(1e-300)
    valid = (
        (squared_edit_norm <= squared_edit_norm_limit)
        & (abs(squared_edit_norm / squared_edit_norm_limit - 1) <= 0.01)
        & (cosine >= 0.9)
    )
    return squared_edit_norm, cosine, valid


@torch.no_grad()
def match_quantized_row(state, direction, squared_edit_norm_limit, maximum_fixed=64):
    base = state.double()
    unit = direction.double() / direction.double().norm()
    held = torch.zeros_like(state, dtype=torch.bool)
    values = state.clone()
    target = 0.995 * squared_edit_norm_limit

    def candidate(alpha):
        translated = (base + alpha * unit).to(state.dtype)
        return torch.where(held, values, translated)

    def squared_edit_norm(updated):
        return float((updated.double() - base).square().sum())

    for fixed_count in range(min(maximum_fixed, state.numel()) + 1):
        low, high = (0.0, 2 * math.sqrt(squared_edit_norm_limit))
        for _ in range(32):
            if squared_edit_norm(candidate(high)) > target:
                break
            high *= 2
        else:
            raise ValueError("Cannot bracket a feasible bf16 dense edit.")
        for _ in range(48):
            middle = (low + high) / 2
            if squared_edit_norm(candidate(middle)) > target:
                high = middle
            else:
                low = middle
        under, over = (candidate(low), candidate(high))
        _, _, valid = geometry(state[None], under[None], direction, squared_edit_norm_limit)
        displacement = under.double() - base
        if bool(valid.all()):
            if not torch.equal(under[direction == 0], state[direction == 0]) or bool(
                (displacement * unit < 0).any()
            ):
                raise ValueError("The bf16 dense edit changed its support or displacement sign.")
            return under, fixed_count
        if fixed_count == min(maximum_fixed, state.numel()):
            break
        jump = (over.double() - base).square() - displacement.square()
        jump = torch.where(held | (direction == 0), torch.zeros_like(jump), jump)
        coordinate = int(jump.argmax())
        if float(jump[coordinate]) <= 0:
            break
        values[coordinate] = under[coordinate]
        held[coordinate] = True
    raise ValueError("No bf16 dense edit satisfies the squared edit norm and cosine limits.")


@torch.no_grad()
def realize_dense(states, direction, squared_edit_norm_limit):
    """Match a bf16 additive edit while preserving zero coordinates and signs."""
    try:
        return _realize_dense(states, direction, squared_edit_norm_limit)
    except ValueError as error:
        if str(error) != "Dense realized squared edit norm and cosine audit failed.":
            raise
    updated, _, _, _, rounding_adjustments = realize_quantized_addition(
        states, direction, math.sqrt(0.995 * squared_edit_norm_limit)
    )
    before_squared_edit_norm, before_cosine, valid = geometry(
        states, updated, direction, squared_edit_norm_limit
    )
    bad_rows = torch.nonzero(~valid, as_tuple=True)[0]
    fixed_counts = []
    for index in bad_rows.tolist():
        corrected, fixed = match_quantized_row(states[index], direction, squared_edit_norm_limit)
        updated[index] = corrected
        fixed_counts.append(fixed)
        rounding_adjustments[index] += fixed
    squared_edit_norm, cosine, valid = geometry(states, updated, direction, squared_edit_norm_limit)
    if not bool(torch.isfinite(updated).all()) or not bool(valid.all()):
        raise ValueError("The bf16 dense edit violates a required numerical limit.")
    QUANTIZATION_DIAGNOSTICS.append(
        dict(
            squared_edit_norm_limit=squared_edit_norm_limit,
            rows=bad_rows.tolist(),
            fixed_coordinates=fixed_counts,
            original_squared_edit_norm_ratio=(
                before_squared_edit_norm[bad_rows] / squared_edit_norm_limit
            )
            .cpu()
            .tolist(),
            original_cosine=before_cosine[bad_rows].cpu().tolist(),
            final_squared_edit_norm_ratio=(squared_edit_norm[bad_rows] / squared_edit_norm_limit)
            .cpu()
            .tolist(),
            final_cosine=cosine[bad_rows].cpu().tolist(),
        )
    )
    unit = direction.double() / direction.double().norm()
    delta = updated.double() - states.double()
    return updated, dict(
        realized_squared_edit_norm=squared_edit_norm,
        requested_squared_edit_norm=torch.full_like(squared_edit_norm, squared_edit_norm_limit),
        squared_edit_norm_error=abs(squared_edit_norm / squared_edit_norm_limit - 1),
        cosine=cosine,
        norm_before=states.double().norm(dim=-1),
        norm_after=updated.double().norm(dim=-1),
        effective_projection=(delta * unit).sum(-1),
        quantization_adjustment_steps=rounding_adjustments.double(),
    )
