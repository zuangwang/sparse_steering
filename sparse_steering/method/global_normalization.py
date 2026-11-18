"""Normalize the entire activation after a dense or sparse edit."""

from __future__ import annotations

import torch


@torch.no_grad()
def realize_global_normalization(states, direction, squared_edit_norm_limits, *, max_steps=40):
    """Match the requested displacement along the globally normalized ray."""
    if states.ndim != 2 or states.dtype != torch.bfloat16 or (not len(states)):
        raise ValueError("States must be a nonempty bf16 matrix.")
    v = torch.as_tensor(direction, device=states.device, dtype=torch.float64)
    squared_edit_norm_limits = torch.as_tensor(
        squared_edit_norm_limits, device=states.device, dtype=torch.float64
    )
    if squared_edit_norm_limits.ndim == 0:
        squared_edit_norm_limits = squared_edit_norm_limits.expand(len(states)).clone()
    if v.shape != (states.shape[1],) or squared_edit_norm_limits.shape != (len(states),):
        raise ValueError("The direction or squared edit norm limits have the wrong shape.")
    if (
        not bool(
            torch.isfinite(states).all()
            & torch.isfinite(v).all()
            & torch.isfinite(squared_edit_norm_limits).all()
        )
        or bool((squared_edit_norm_limits < 0).any())
        or float(v.norm()) <= 1e-12
    ):
        raise ValueError(
            "States must be finite, the direction nonzero "
            "and the squared edit norm limits nonnegative."
        )
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("Positive integer backoff count required.")
    v = v / v.norm()
    h = states.double()
    radius = h.norm(dim=-1)
    squared = radius.square()
    projection = (h * v).sum(-1)
    cosine = (projection / radius.clamp_min(1e-300)).clamp(-1, 1)
    sine = (1 - cosine.square()).clamp_min(0).sqrt()
    active = (radius > 0) & (sine > 1e-10) & (squared_edit_norm_limits > 0)
    limit = 2 * squared * (1 - cosine)
    finite_limit = limit * (1 - 1e-06)
    target = torch.minimum(squared_edit_norm_limits, finite_limit)
    t = (target / (2 * squared).clamp_min(1e-300)).clamp(0, 2)
    cos_angle = 1 - t
    sin_angle = (t * (2 - t)).clamp_min(0).sqrt()
    denominator = sine * cos_angle - cosine * sin_angle
    nominal_alpha = radius * sin_angle / denominator.clamp_min(1e-300)
    nominal_alpha = torch.where(active, nominal_alpha, 0.0)
    if not bool(torch.isfinite(nominal_alpha).all()) or bool((active & (denominator <= 0)).any()):
        raise ValueError("Invalid connected normalization-ray endpoint.")

    def candidate(alpha):
        intermediate = h + alpha[:, None] * v
        norm = intermediate.norm(dim=-1)
        beta = torch.where(radius > 0, radius / norm.clamp_min(1e-300), 1.0)
        ideal = intermediate * beta[:, None]
        updated = torch.where((alpha == 0)[:, None], states, ideal.to(states.dtype))
        actual = updated.double() - h
        planned = ideal - h
        squared_edit_norm = actual.square().sum(-1)
        ideal_squared_edit_norm = planned.square().sum(-1)
        after = updated.double().norm(dim=-1)
        drift = torch.where(radius > 0, (after / radius.clamp_min(1e-300) - 1).abs(), after)
        chord_cosine = (actual * planned).sum(-1) / (
            squared_edit_norm * ideal_squared_edit_norm
        ).sqrt().clamp_min(1e-300)
        chord_cosine = torch.where(squared_edit_norm == 0, 1.0, chord_cosine)
        valid = (
            (squared_edit_norm <= squared_edit_norm_limits)
            & (drift <= 0.001)
            & (chord_cosine >= 0.9)
            & torch.isfinite(updated).all(-1)
            & ((norm > 0) | (radius == 0))
        )
        return (
            updated,
            valid,
            squared_edit_norm,
            ideal_squared_edit_norm,
            drift,
            chord_cosine,
            beta,
            after,
        )

    # Back off from the ideal coefficient if the bf16 displacement exceeds
    # its limit or fails the norm and chord checks.
    trial = candidate(nominal_alpha)
    best_alpha = torch.where(trial[1], nominal_alpha, 0.0)
    searching = active & ~trial[1]
    low, high = (torch.zeros_like(radius), nominal_alpha.clone())
    if bool(searching.any()):
        for _ in range(max_steps):
            mid = (low + high) / 2
            current = candidate(mid)
            better = searching & current[1] & (mid > best_alpha)
            best_alpha = torch.where(better, mid, best_alpha)
            low = torch.where(searching & current[1], mid, low)
            high = torch.where(searching & ~current[1], mid, high)
    final = candidate(best_alpha)
    best_alpha = torch.where(final[2] == 0, 0.0, best_alpha)
    final = candidate(best_alpha)
    if not bool(final[1].all()):
        raise ValueError(
            "The globally normalized edit violates a displacement, norm or chord limit."
        )
    updated, _, squared_edit_norm, ideal_squared_edit_norm, drift, chord_cosine, beta, after = final
    delta = updated.double() - h
    support = v != 0
    inside = delta[:, support].square().sum(-1)
    outside = delta[:, ~support].square().sum(-1)
    diagnostics = dict(
        realized_squared_edit_norm=squared_edit_norm,
        ideal_squared_edit_norm=ideal_squared_edit_norm,
        squared_edit_norm_limit=squared_edit_norm_limits,
        norm_relative_drift=drift,
        planned_chord_cosine=chord_cosine,
        alpha=best_alpha,
        beta=beta,
        nominal_alpha=nominal_alpha,
        activation_norm_before=radius,
        activation_norm_after=after,
        projection=projection,
        physical_squared_edit_norm_limit=limit,
        capacity_limited=(squared_edit_norm_limits > finite_limit)
        | (squared_edit_norm_limits > 0) & ~active,
        numerical_backoff=best_alpha < nominal_alpha,
        no_op=squared_edit_norm == 0,
        inside_support_squared_edit_norm=inside,
        outside_support_squared_edit_norm=outside,
        changed_coordinates=(delta != 0).sum(-1),
        steering_coordinate_count=int(support.sum()),
    )
    return updated, diagnostics


def make_hook(runtime, torch, direction):
    from sparse_steering.method.constrained_normalization import realize_constrained_normalization

    records = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        states = hidden[:, -1, :]
        assert states.dtype == torch.bfloat16
        squared_edit_norm_limit = (
            runtime.cell["edit_norm_ratio_squared"]
            * runtime.scale**2
            * (
                1
                if runtime._prediction_step == 0
                else runtime.cell["continuation_squared_norm_multiplier"]
            )
        )
        updated, raw = realize_constrained_normalization(
            states, direction, ~(direction != 0), squared_edit_norm_limit
        )
        squared_edit_norm = (updated.double() - states.double()).square().sum(-1)
        torch.testing.assert_close(
            squared_edit_norm, raw["realized_squared_edit_norm"], rtol=0, atol=1e-10
        )
        if not bool(
            (
                (squared_edit_norm <= squared_edit_norm_limit)
                & (raw["norm_relative_drift"] <= 0.001)
                & (raw["planned_chord_cosine"] >= 0.9)
                & torch.isfinite(updated).all(-1)
            ).all()
        ):
            raise ValueError(
                "The globally normalized edit violates a displacement, norm, "
                "cosine or finite-value limit."
            )
        diag = dict(
            realized_squared_edit_norm=squared_edit_norm,
            requested_squared_edit_norm=torch.full_like(squared_edit_norm, squared_edit_norm_limit),
            cosine=raw["planned_chord_cosine"],
            norm_relative_drift=raw["norm_relative_drift"],
            outside_support_squared_edit_norm=raw["outside_support_squared_edit_norm"],
            beta=raw["beta"],
            alpha=raw["alpha"],
            effective_target_squared_edit_norm=raw["effective_target_squared_edit_norm"],
            physical_squared_edit_norm_limit=raw["physical_squared_edit_norm_limit"],
            target_relative_error=raw["target_relative_error"],
            capacity_limited=raw["capacity_limited"].double(),
            numerical_backoff=raw["numerical_backoff"].double(),
            no_op=raw["no_op"].double(),
            changed_coordinates=raw["changed_coordinates"].double(),
            support_unchanged=raw["support_unchanged"].double(),
        )
        records.append(diag)
        names = list(diag)
        values = torch.stack([diag[k].double() for k in names], -1).cpu().tolist()
        for context, value in zip(runtime._contexts, values):
            runtime.prediction_trace.append(
                dict(
                    context,
                    prediction_step=runtime._prediction_step,
                    **dict(zip(names, value)),
                    steering_coordinate_count=raw["steering_coordinate_count"],
                    normalization_coordinates=raw["normalization_support_count"],
                )
            )
        runtime._prediction_step += 1
        changed = hidden.clone()
        changed[:, -1, :] = updated
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    return hook, records
