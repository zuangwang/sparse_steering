"""Select the highest score within the paper's text-quality limits."""

import math


def best_feasible(points):
    return min(
        (point for point in points if point["feasible"]),
        key=lambda point: (
            -point["score"],
            point["actual_squared_edit_norm"],
            point["edit_norm_ratio_squared"],
        ),
        default=None,
    )


def select(grid, *, nll_ratio_limit=1.15, repetition_limit=0.5):
    (base,) = [point for point in grid if point["edit_norm_ratio_squared"] == 0]
    assert base["completion_nll"] > 0
    assert all(
        (
            math.isfinite(point[key])
            for point in grid
            for key in ("score", "actual_squared_edit_norm", "completion_nll", "repeated_4gram")
        )
    )
    feasible = [
        point
        for point in grid
        if point["completion_nll"] / base["completion_nll"] <= nll_ratio_limit + 1e-12
        and point["repeated_4gram"] <= repetition_limit + 1e-12
    ]
    if not feasible:
        return dict(available=False, reason="No setting satisfies the text-quality limits.")
    best = min(
        feasible,
        key=lambda point: (
            -point["score"],
            point["actual_squared_edit_norm"],
            point["edit_norm_ratio_squared"],
        ),
    )
    return dict(best, available=True, nll_ratio=best["completion_nll"] / base["completion_nll"])
