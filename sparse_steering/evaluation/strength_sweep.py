"""Configure each method on the common steering-strength grid."""

from __future__ import annotations

import math


def configure(runtime, torch, original, method, edit_norm_ratio_squared, fit):
    if method in ("ours", "sparsification_only", "sparsification_global_norm"):
        from sparse_steering.runtime.hooks import install_sparse as install

        install(runtime, torch)
        config = dict(
            method=method if edit_norm_ratio_squared else "unsteered",
            edit_norm_ratio_squared=edit_norm_ratio_squared,
            continuation_squared_norm_multiplier=continuation_squared_norm_multiplier(
                edit_norm_ratio_squared
            ),
        )
        vector = fit["masked_direction"]
    elif method in ("magnitude", "gradient", "ie"):
        from sparse_steering.runtime.hooks import install_coordinate as install

        install(runtime, torch)
        config = dict(
            method="sparsification_only" if edit_norm_ratio_squared else "unsteered",
            edit_norm_ratio_squared=edit_norm_ratio_squared,
            continuation_squared_norm_multiplier=continuation_squared_norm_multiplier(
                edit_norm_ratio_squared
            ),
        )
        vector = fit["direction"]
    else:
        runtime.make_hook = original
        config = cell(method, edit_norm_ratio_squared)
        vector = fit["normalized_dense"] if method == "normalized_dense" else fit["dense"]
    runtime.configure(config, fit["scale"])
    runtime.extra_normalization_mask = torch.tensor(
        fit.get("extra_normalization_mask", [0] * len(vector)), device="cuda", dtype=torch.bool
    )
    return config, (
        torch.tensor(vector, device="cuda", dtype=torch.float64)
        if edit_norm_ratio_squared
        else None
    )


def continuation_squared_norm_multiplier(edit_norm_ratio_squared):
    """Scale the requested squared norm after the first prediction position."""
    if not math.isfinite(edit_norm_ratio_squared) or edit_norm_ratio_squared < 0:
        raise ValueError("Squared edit norm ratio must be finite and nonnegative.")
    return 1.0 if edit_norm_ratio_squared else 0.0


def cell(method, edit_norm_ratio_squared):
    op = (
        "unsteered"
        if edit_norm_ratio_squared == 0
        else "normalized_dense" if method == "normalized_dense" else "dense"
    )
    return dict(
        method=op,
        edit_norm_ratio_squared=float(edit_norm_ratio_squared),
        continuation_squared_norm_multiplier=0.0 if not edit_norm_ratio_squared else 1.0,
        config_id=(
            "unsteered"
            if not edit_norm_ratio_squared
            else f"{op}_b{edit_norm_ratio_squared:.6f}".replace(".", "p")
        ),
    )


def config(method, edit_norm_ratio_squared):
    if method == "unsteered" or edit_norm_ratio_squared == 0:
        return dict(
            method="unsteered",
            edit_norm_ratio_squared=0.0,
            continuation_squared_norm_multiplier=0.0,
            config_id="unsteered",
        )
    return dict(
        method=method,
        edit_norm_ratio_squared=float(edit_norm_ratio_squared),
        continuation_squared_norm_multiplier=min(1 / 32, 0.04 / (32 * edit_norm_ratio_squared)),
        config_id=f"{method}_b{edit_norm_ratio_squared:.4f}".replace(".", "p"),
    )


NUMERICAL_FAILURES = {
    "No bf16 dense edit satisfies the squared edit norm and cosine limits.",
    "Cannot bracket a feasible bf16 dense edit.",
    "The bf16 dense edit violates a required numerical limit.",
}
