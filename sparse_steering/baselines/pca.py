"""Fit the dense and sparse principal-component baselines."""

import numpy as np


def fit_arrays(states, pairs, width, interleaved):
    states = np.asarray(states, dtype=np.float64)
    if states.shape != (2 * pairs, width) or not np.isfinite(states).all():
        raise ValueError("The training activation panel has the wrong shape or nonfinite values.")
    positive, negative = (states[::2], states[1::2]) if interleaved else np.split(states, 2)
    difference = positive - negative
    centered = difference - difference.mean(0, keepdims=True)
    # Solve the smaller paired-example Gram matrix, then lift its top vector.
    values, vectors = np.linalg.eigh(centered @ centered.T)
    if values[-1] <= 1e-12:
        raise ValueError("Degenerate paired PCA.")
    pc = centered.T @ vectors[:, -1]
    pc /= np.linalg.norm(pc)
    if pc[np.argmax(np.abs(pc))] < 0:
        pc = -pc
    # Fix the arbitrary eigenvector sign using the preferred-answer majority.
    majority = np.mean(difference @ pc > 0) - np.mean(difference @ pc < 0)
    if majority < 0:
        pc = -pc
    pc = pc.astype(np.float32)
    k = int(width * 0.05)
    indices = np.lexsort((np.arange(width), -np.abs(pc)))[:k]
    sparse = np.zeros_like(pc)
    sparse[indices] = pc[indices]
    if not 0 < np.linalg.norm(sparse) or np.count_nonzero(sparse) != k:
        raise ValueError("Degenerate top-five-percent PC.")
    raw = positive.mean(0) - negative.mean(0)
    raw /= np.linalg.norm(raw)
    return (
        dict(pca_dense=pc.tolist(), pca_sparse=sparse.tolist()),
        dict(
            pairs=pairs,
            ordering="interleaved" if interleaved else "positive_then_negative",
            centered_paired_pc=True,
            explained_variance_ratio=float(values[-1] / values.sum()),
            positive_projection_fraction=float(np.mean(difference @ pc > 0)),
            sparse_count=k,
            sparse_fraction=k / width,
        ),
        raw.astype(np.float32),
        float(np.linalg.norm(states, axis=1).mean()),
    )
