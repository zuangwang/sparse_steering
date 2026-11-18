"""Score the fixed WikiText context and continuation windows."""

import collections
import math

CONTEXT = 256
SCORED = 128


def score(runtime, keys, sequences):
    torch = runtime.torch
    tokens = torch.tensor(sequences, device="cuda", dtype=torch.long)
    handle = None
    with torch.no_grad():
        try:
            handle = runtime.begin(keys)
            # The last context position predicts the first scored token.
            # Continue with the observed tokens while retaining the steered cache.
            ids = tokens[:, :CONTEXT]
            mask = torch.ones_like(ids)
            past = None
            losses = []
            for pos in range(SCORED):
                result = runtime.forward(
                    ids, mask, past_key_values=past, use_cache=True, logits_to_keep=1
                )
                logits = result.logits[:, -1, :].float()
                losses.append(
                    torch.nn.functional.cross_entropy(
                        logits, tokens[:, CONTEXT + pos], reduction="none"
                    )
                )
                past = result.past_key_values
                ids = tokens[:, CONTEXT + pos : CONTEXT + pos + 1]
                mask = torch.cat((mask, torch.ones_like(ids)), 1)
            losses = torch.stack(losses, 1).double().cpu()
            del past, result
        finally:
            if handle is not None:
                handle.remove()
    if not torch.isfinite(losses).all():
        raise ValueError("Nonfinite reference loss.")
    traces = runtime.original.prediction_trace
    by_id = collections.defaultdict(list)
    for row in traces:
        by_id[row["id"]].append(row)
    values = []
    for key, loss in zip(keys, losses):
        trace = by_id[str(key)]
        if runtime.direction is not None:
            if sorted((r["prediction_step"] for r in trace)) != list(range(SCORED)):
                raise ValueError("Steering coverage differs across prediction positions.")
            for row in trace:
                expected = (
                    runtime.original.cell["edit_norm_ratio_squared"]
                    * runtime.original.scale**2
                    * (
                        1
                        if row["prediction_step"] == 0
                        else runtime.original.cell["continuation_squared_norm_multiplier"]
                    )
                )
                if not math.isclose(row["requested_squared_edit_norm"], expected, rel_tol=1e-12):
                    raise ValueError("Position squared edit norm schedule differs.")
                if row["realized_squared_edit_norm"] > expected:
                    raise ValueError(
                        "A scoring edit exceeds the requested squared edit norm limit."
                    )
        elif trace:
            raise ValueError("The unsteered evaluation contains a steering edit.")
        values.append(
            dict(
                item_id=str(key),
                nll_sum=float(loss.sum()),
                scored_tokens=SCORED,
                initial_nll_sum=float(loss[:32].sum()),
                initial_scored_tokens=32,
                prediction_step_count=len(trace),
                actual_squared_edit_norm=sum((r["realized_squared_edit_norm"] for r in trace))
                / runtime.original.scale**2,
                max_norm_drift=max((r["norm_relative_drift"] for r in trace), default=0.0),
            )
        )
    return values, traces, losses
