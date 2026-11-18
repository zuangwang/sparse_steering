"""Compute paired bootstrap intervals for capability changes."""

import numpy as np


def paired(values, unsteered, benchmark, seed=0):
    """Resample aligned item differences to compute a paired confidence interval."""
    key = "item_id" if benchmark == "wikitext" else "id"
    lookup = {str(v[key]): v for v in unsteered}
    assert len(lookup) == len(unsteered) == len(values)
    assert {str(v[key]) for v in values} == set(lookup)
    if benchmark == "wikitext":
        delta = np.array(
            [
                v["nll_sum"] / v["scored_tokens"]
                - lookup[str(v[key])]["nll_sum"] / lookup[str(v[key])]["scored_tokens"]
                for v in values
            ]
        )
        groups = [np.arange(len(values))]
    else:
        delta = np.array([v["accuracy"] - lookup[str(v[key])]["accuracy"] for v in values])
        groups = (
            [
                np.array([i for i, v in enumerate(values) if v["subject"] == subject])
                for subject in sorted({v["subject"] for v in values})
            ]
            if benchmark == "mmlu"
            else [np.arange(len(values))]
        )
    # Resample aligned per-item differences; both conditions share each draw.
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(40):
        block = np.zeros(250)
        for indices in groups:
            sample = rng.integers(0, len(indices), size=(250, len(indices)))
            block += delta[indices][sample].mean(1) / len(groups)
        draws.extend(block.tolist())
    estimate = float(sum((delta[g].mean() for g in groups)) / len(groups))
    return dict(
        delta=estimate,
        ci95=np.percentile(draws, [2.5, 97.5], method="linear").tolist(),
        seed=seed,
        resamples=10000,
        unit="nats/token" if benchmark == "wikitext" else "accuracy fraction",
    )
