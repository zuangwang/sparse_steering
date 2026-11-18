"""Check sweep membership, prediction coverage and realized edit geometry."""

import math

import numpy as np


def raw_audit(values, prediction_steps, rows, kind, config, fit):
    if config["method"] == "sparsification_global_norm":
        return raw_audit(
            values,
            prediction_steps,
            rows,
            kind,
            dict(config, method="ours"),
            dict(fit, extra_normalization_mask=[v == 0 for v in fit["masked_direction"]]),
        )
    binary = kind == "binary"
    key = lambda v: (v["id"], v["order"]) if binary else (v["id"],)
    expected = (
        {(r["id"], o) for r in rows for o in (0, 1)} if binary else {(r["id"],) for r in rows}
    )
    assert len(expected) == len(rows) * (2 if binary else 1)
    assert len(values) == len(expected) and {key(v) for v in values} == expected
    n = 1 if binary else 32
    prediction_keys = [(*key(a), a["prediction_step"]) for a in prediction_steps]
    wanted = (
        {(*k, i) for k in expected for i in range(n)}
        if config["edit_norm_ratio_squared"]
        else set()
    )
    assert len(prediction_keys) == len(wanted) and set(prediction_keys) == wanted
    totals = {k: 0.0 for k in expected}
    requests = dict(totals)
    steering_count = sum((v != 0 for v in fit["masked_direction"]))
    support = sum(
        (v != 0 or b for v, b in zip(fit["masked_direction"], fit["extra_normalization_mask"]))
    )
    scaling = config["method"] == "ours"
    for a in prediction_steps:
        assert all((math.isfinite(v) for v in a.values() if isinstance(v, (int, float))))
        squared_edit_norm_limit = (
            config["edit_norm_ratio_squared"]
            * fit["scale"] ** 2
            * (1 if a["prediction_step"] == 0 else config["continuation_squared_norm_multiplier"])
        )
        assert (
            a["requested_squared_edit_norm"] == squared_edit_norm_limit
            and 0 <= a["realized_squared_edit_norm"] <= squared_edit_norm_limit
        )
        assert (
            a["cosine"] >= 0.9
            and a["outside_support_squared_edit_norm"] == 0
            and (a["support_unchanged"] == 1)
        )
        assert a["changed_coordinates"] <= (support if scaling else steering_count)
        if scaling:
            assert a["norm_relative_drift"] <= 0.001 and a["alpha"] >= 0 and (a["beta"] > 0)
            assert (
                a["steering_coordinate_count"] == steering_count
                and a["normalization_coordinates"] == support
            )
            assert 0 <= a["effective_target_squared_edit_norm"] <= squared_edit_norm_limit
            assert bool(a["no_op"]) == (a["realized_squared_edit_norm"] == 0)
        else:
            assert (
                abs(a["realized_squared_edit_norm"] / squared_edit_norm_limit - 1) <= 0.01
                and a["displacement_sign_aligned"] == 1
            )
        totals[key(a)] += a["realized_squared_edit_norm"]
        requests[key(a)] += a["requested_squared_edit_norm"]
    for v in values:
        assert v["prediction_step_count"] == (n if config["edit_norm_ratio_squared"] else 0)
        assert math.isclose(
            v["realized_squared_edit_norm"], totals[key(v)], rel_tol=1e-12, abs_tol=1e-08
        )
        assert math.isclose(
            v["requested_squared_edit_norm"], requests[key(v)], rel_tol=1e-12, abs_tol=1e-08
        )
        if binary:
            assert v["accuracy"] in (0.0, 0.5, 1.0)
            assert all(
                (
                    math.isfinite(v[k])
                    for k in (
                        "correct_probability",
                        "answer_label_probability_mass",
                        "correct_answer_label_nll",
                    )
                )
            )
        else:
            words = v["text"].split()
            grams = [tuple(words[j : j + 4]) for j in range(max(0, len(words) - 3))]
            assert v["repeated_4gram"] == int(len(grams) != len(set(grams)))
            assert math.isfinite(v["completion_nll"]) and v["completion_nll"] >= 0
    metrics = dict(
        actual_squared_edit_norm=sum((v["realized_squared_edit_norm"] for v in values))
        / (len(values) * fit["scale"] ** 2)
    )
    if binary:
        metrics["score"] = sum((v["accuracy"] for v in values)) / len(values)
    else:
        metrics["generation_squared_edit_norm"] = metrics.pop("actual_squared_edit_norm")
        metrics.update(
            {
                k: sum((v[k] for v in values)) / len(values)
                for k in ("completion_nll", "repeated_4gram")
            }
        )
    return dict(
        passed=True,
        prediction_steps=len(prediction_steps),
        questions=len(rows),
        metrics=metrics,
        maximum_norm_drift=max((a["norm_relative_drift"] for a in prediction_steps), default=0.0),
        minimum_cosine=min((a["cosine"] for a in prediction_steps), default=1.0),
        maximum_squared_edit_norm_ratio=max(
            (
                a["realized_squared_edit_norm"] / a["requested_squared_edit_norm"]
                for a in prediction_steps
            ),
            default=0.0,
        ),
        capacity_limited_prediction_steps=sum(
            (bool(a.get("capacity_limited")) for a in prediction_steps)
        ),
        numerical_backoff_prediction_steps=sum(
            (bool(a.get("numerical_backoff")) for a in prediction_steps)
        ),
        no_op_prediction_steps=sum(
            (a["realized_squared_edit_norm"] == 0 for a in prediction_steps)
        ),
        adjusted_prediction_steps=sum(
            (a.get("quantization_adjustment_steps", 0) > 0 for a in prediction_steps)
        ),
    )


def audit(values, trace, rows, evaluation, cell, scale):
    keys = ["id", "order"] if evaluation == "binary" else ["id"]
    expected = (
        {(r["id"], o) for r in rows for o in (0, 1)}
        if evaluation == "binary"
        else {(r["id"],) for r in rows}
    )
    actual_keys = [tuple((v[k] for k in keys)) for v in values]
    if set(actual_keys) != expected or len(actual_keys) != len(expected):
        raise ValueError("Incomplete or duplicated evaluation panel.")
    if cell["method"] == "unsteered":
        if trace or any(
            (
                v["realized_squared_edit_norm"] != 0 or v["prediction_step_count"] != 0
                for v in values
            )
        ):
            raise ValueError("The unsteered setting contains a steering edit.")
        return dict(
            passed=True,
            prediction_steps=0,
            maximum_squared_edit_norm_ratio=0.0,
            maximum_norm_drift=0.0,
        )
    count = 1 if evaluation == "binary" else 32
    trace_keys = [tuple((r[k] for k in keys)) + (r["prediction_step"],) for r in trace]
    if len(trace_keys) != len(expected) * count or set(trace_keys) != {
        k + (a,) for k in expected for a in range(count)
    }:
        raise ValueError("The trace is incomplete at the prediction positions.")
    maximum_squared_edit_norm_ratio, maximum_norm_drift, minimum_cosine, beta_max = (
        0.0,
        0.0,
        1.0,
        0.0,
    )
    sums = {k: 0.0 for k in expected}
    for t in trace:
        squared_edit_norm_limit = (
            cell["edit_norm_ratio_squared"]
            * scale**2
            * (1 if t["prediction_step"] == 0 else cell["continuation_squared_norm_multiplier"])
        )
        if t["requested_squared_edit_norm"] != squared_edit_norm_limit or not all(
            (np.isfinite(v) for k, v in t.items() if k != "id")
        ):
            raise ValueError(
                "The schedule changed or a prediction record contains nonfinite values."
            )
        ratio = t["realized_squared_edit_norm"] / squared_edit_norm_limit
        if not 0 <= ratio <= 1 or t["cosine"] < 0.9:
            raise ValueError("The edit violates its displacement or cosine limit.")
        if cell["method"] in ("dense",) and abs(ratio - 1) > 0.01:
            raise ValueError("Additive matched squared edit norm gate failed.")
        if cell["method"] in ("ours", "normalized_dense") and t["norm_relative_drift"] > 0.001:
            raise ValueError("Norm preservation failed.")
        if cell["method"] == "ours" and (
            t["outside_support_squared_edit_norm"] != 0 or t["beta"] < 0
        ):
            raise ValueError("Sparse scaled support or feasibility failed.")
        sums[tuple((t[k] for k in keys))] += t["realized_squared_edit_norm"]
        maximum_squared_edit_norm_ratio = max(maximum_squared_edit_norm_ratio, ratio)
        maximum_norm_drift = max(maximum_norm_drift, t["norm_relative_drift"])
        minimum_cosine = min(minimum_cosine, t["cosine"])
        beta_max = max(beta_max, t["beta"])
    for v in values:
        if v["prediction_step_count"] != count or not np.isclose(
            v["realized_squared_edit_norm"],
            sums[tuple((v[k] for k in keys))],
            rtol=1e-12,
            atol=1e-10,
        ):
            raise ValueError("Aggregate values differ from the per-position records.")
    return dict(
        passed=True,
        prediction_steps=len(trace),
        maximum_squared_edit_norm_ratio=maximum_squared_edit_norm_ratio,
        maximum_norm_drift=maximum_norm_drift,
        minimum_cosine=minimum_cosine,
        maximum_beta=beta_max,
        capacity_limited_fraction=float(np.mean([t["capacity_limited"] for t in trace])),
        numerical_backoff_fraction=float(np.mean([t["numerical_backoff"] for t in trace])),
        no_op_fraction=float(np.mean([t["no_op"] for t in trace])),
    )
