"""Restore the activation norm by rescaling only the selected coordinates."""

from __future__ import annotations

import torch


def _validate(states, direction, squared_edit_norms):
    if states.ndim != 2 or states.dtype != torch.bfloat16 or (not len(states)):
        raise ValueError("States must be a nonempty BF16 matrix.")
    direction = torch.as_tensor(direction, device=states.device, dtype=torch.float64)
    if direction.ndim != 1 or direction.numel() != states.shape[1]:
        raise ValueError("Direction must match hidden dimension.")
    squared_edit_norms = torch.as_tensor(
        squared_edit_norms, device=states.device, dtype=torch.float64
    )
    if squared_edit_norms.ndim == 0:
        squared_edit_norms = squared_edit_norms.expand(len(states)).clone()
    if squared_edit_norms.shape != (len(states),):
        raise ValueError("Squared edit norm must be a scalar or one value per row.")
    if not bool(
        torch.isfinite(states).all()
        & torch.isfinite(direction).all()
        & torch.isfinite(squared_edit_norms).all()
    ) or bool((squared_edit_norms < 0).any()):
        raise ValueError(
            "States, directions and squared edit norms must be finite; "
            "squared edit norms must also be nonnegative."
        )
    direction_norm = direction.norm()
    if float(direction_norm) <= 1e-12:
        raise ValueError("Direction must be nonzero.")
    return direction / direction_norm, squared_edit_norms


@torch.no_grad()
def realize_constrained_normalization(
    states, direction, extra_normalization_mask, squared_edit_norm_limits
):
    """Add the sparse direction and restore the norm on its selected support.

    Search the additive multiplier so the final rescaled edit is no larger
    than the dense edit requested at this strength. Compute candidates in
    float64, then audit their actual bf16 displacement and norm drift.
    """
    v, squared_edit_norm_limits = _validate(states, direction, squared_edit_norm_limits)
    extra_normalization_mask = torch.as_tensor(
        extra_normalization_mask, device=states.device, dtype=torch.bool
    )
    if extra_normalization_mask.shape != v.shape or bool(
        ((v != 0) & extra_normalization_mask).any()
    ):
        raise ValueError("Scaling support must be disjoint from the frozen steering direction.")
    # Extra normalization coordinates have zero additive direction. The union
    # is rescaled; its complement remains bitwise unchanged.
    support = (v != 0) | extra_normalization_mask
    x = states[:, support].double()
    full_before = states.double().norm(dim=-1)
    outside_squared = states[:, ~support].double().square().sum(-1)
    u = v[support]
    r = x.norm(dim=-1)
    r2 = r.square()
    q = (x * u).sum(-1)
    c = (q / r.clamp_min(1e-300)).clamp(-1, 1)
    s = (1 - c.square()).clamp_min(0).sqrt()
    active = (r > 0) & (s > 1e-10) & (squared_edit_norm_limits > 0)
    # The connected nonnegative ray has a finite chord-length limit.
    limit = torch.where(active, 2 * r2 * (1 - c), 0.0)
    finite_limit = limit * (1 - 1e-06)
    target = torch.minimum(squared_edit_norm_limits, finite_limit)
    max_t = torch.where(active, (1 - c) * (1 - 1e-06), 0.0)
    nominal_t = torch.where(active, target / (2 * r2).clamp_min(1e-300), 0.0)

    def candidate(t):
        sn = (t * (2 - t)).clamp_min(0).sqrt()
        den = s * (1 - t) - c * sn
        alpha = torch.where(active, r * sn / den.clamp_min(1e-300), 0.0)
        # Paper sparse-edit equation, restricted to the normalization support.
        intermediate = x + alpha[:, None] * u
        beta = torch.where(r > 0, r / intermediate.norm(dim=-1).clamp_min(1e-300), 1.0)
        # Paper constrained-normalization equation: beta restores this norm.
        ideal = beta[:, None] * intermediate
        # Validate the single bf16 cast, rather than the ideal float64 update.
        out = torch.where((alpha == 0)[:, None], states[:, support], ideal.bfloat16())
        delta = out.double() - x
        planned = ideal - x
        squared_edit_norm = delta.square().sum(-1)
        ideal_squared_edit_norm = planned.square().sum(-1)
        after = out.double().norm(dim=-1)
        drift = torch.where(r > 0, (after / r.clamp_min(1e-300) - 1).abs(), after)
        full_after = (outside_squared + after.square()).sqrt()
        full_drift = torch.where(
            full_before > 0, (full_after / full_before.clamp_min(1e-300) - 1).abs(), full_after
        )
        cosine = (delta * planned).sum(-1) / (
            squared_edit_norm * ideal_squared_edit_norm
        ).sqrt().clamp_min(1e-300)
        cosine = torch.where(squared_edit_norm == 0, 1.0, cosine)
        valid = (
            (squared_edit_norm <= squared_edit_norm_limits)
            & (full_drift <= 0.001)
            & (cosine >= 0.9)
            & torch.isfinite(out).all(-1)
            & torch.isfinite(alpha)
            & (alpha >= 0)
        )
        score = torch.where(valid, (squared_edit_norm - target).abs(), torch.inf)
        return (
            out,
            score,
            squared_edit_norm,
            ideal_squared_edit_norm,
            alpha,
            beta,
            drift,
            cosine,
            after,
        )

    best_t = torch.zeros_like(r)
    best_score = candidate(best_t)[1]

    def consider(t):
        nonlocal best_t, best_score
        trial = candidate(t)
        better = trial[1] < best_score
        best_t = torch.where(better, t, best_t)
        best_score = torch.minimum(best_score, trial[1])
        return trial[2]

    # Quantization makes displacement discontinuous. Keep the closest valid
    # candidate while bisecting, then scan nearby or smaller chords if needed.
    consider(nominal_t)
    low = torch.zeros_like(r)
    high = torch.minimum(max_t, nominal_t * 1.02 + 1e-12)
    for _ in range(28):
        mid = (low + high) / 2
        squared_edit_norm = consider(mid)
        low = torch.where(squared_edit_norm < target, mid, low)
        high = torch.where(squared_edit_norm >= target, mid, high)
    if bool((best_score > target * 0.005).any()):
        for i in range(81):
            consider(torch.minimum(max_t, nominal_t * (0.98 + 0.0005 * i)))
    if bool((best_score > target * 0.01).any()):
        for i in range(1, 65):
            consider(nominal_t * (i / 64))
    out, score, squared_edit_norm, ideal_squared_edit_norm, alpha, beta, drift, cosine, after = (
        candidate(best_t)
    )
    if not bool(torch.isfinite(score).all()):
        raise ValueError(
            "Constrained normalization failed its scalar, norm and displacement gates."
        )
    y = states.clone()
    y[:, support] = out
    full_before, full_after = (states.double().norm(dim=-1), y.double().norm(dim=-1))
    full_drift = torch.where(
        full_before > 0, (full_after / full_before.clamp_min(1e-300) - 1).abs(), full_after
    )
    target_error = torch.where(
        target > 0, (squared_edit_norm - target).abs() / target.clamp_min(1e-300), 0.0
    )
    requested_error = torch.where(
        squared_edit_norm_limits > 0,
        (squared_edit_norm - squared_edit_norm_limits).abs()
        / squared_edit_norm_limits.clamp_min(1e-300),
        0.0,
    )
    diag = dict(
        realized_squared_edit_norm=squared_edit_norm,
        ideal_squared_edit_norm=ideal_squared_edit_norm,
        squared_edit_norm_limit=squared_edit_norm_limits,
        effective_target_squared_edit_norm=target,
        target_relative_error=target_error,
        requested_relative_error=requested_error,
        alpha=alpha,
        beta=beta,
        selected_norm_before=r,
        selected_norm_after=after,
        selected_norm_relative_drift=drift,
        activation_norm_before=full_before,
        activation_norm_after=full_after,
        norm_relative_drift=full_drift,
        projection=q,
        physical_squared_edit_norm_limit=limit,
        capacity_limited=squared_edit_norm_limits > finite_limit,
        numerical_backoff=target_error > 0.01,
        no_op=squared_edit_norm == 0,
        planned_chord_cosine=cosine,
        inside_support_squared_edit_norm=squared_edit_norm,
        outside_support_squared_edit_norm=torch.zeros_like(squared_edit_norm),
        changed_coordinates=(out != states[:, support]).sum(-1),
        support_unchanged=(y[:, ~support] == states[:, ~support]).all(-1),
        steering_coordinate_count=int((v != 0).sum()),
        normalization_support_count=int(support.sum()),
    )
    return y, diag
