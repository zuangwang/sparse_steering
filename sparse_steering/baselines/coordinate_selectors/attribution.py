"""Compute indirect effects on training examples using midpoint attribution."""

import json
import math
import time

from sparse_steering.runtime.storage import SETTINGS, SETTINGS_PATH, digest, freeze, read, save, sha

P = {"attribution": SETTINGS["attribution"]}
PROTOCOL = SETTINGS_PATH


def logits(runtime, torch, ids, start, delta, answer_label_ids=None):
    tokens = torch.tensor([ids], dtype=torch.long, device=next(runtime.model.parameters()).device)
    attention = torch.ones_like(tokens)
    count = len(ids) - start
    if (
        start < 0
        or count < 1
        or count > 32
        or (len(ids) > runtime.model.config.max_position_embeddings)
    ):
        raise ValueError("Invalid attribution response positions.")
    handle = None
    if delta is not None:
        weights = torch.full(
            (count,),
            1 / math.sqrt(32),
            device=next(runtime.model.parameters()).device,
            dtype=delta.dtype,
        )
        weights[0] = 1

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            changed = hidden.clone()
            changed[:, start:, :] = hidden[:, start:, :] + (weights[:, None] * delta[None, :]).to(
                hidden.dtype
            )
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        handle = runtime.layer.register_forward_hook(hook)
    try:
        output = runtime.forward(tokens, attention, use_cache=False, logits_to_keep=count)
        scores = output.logits[0, -count:, :].float()
        if answer_label_ids is not None:
            scores = scores[:, answer_label_ids]
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("Nonfinite attribution logits.")
        return scores
    finally:
        if handle is not None:
            handle.remove()


def generation(runtime, torch, prefix, delta):
    ids = torch.tensor([prefix], dtype=torch.long, device=next(runtime.model.parameters()).device)
    mask = torch.ones_like(ids)
    past, tokens = (None, [])
    prediction_step = 0

    def hook(_module, _inputs, output):
        nonlocal prediction_step
        hidden = output[0] if isinstance(output, tuple) else output
        changed = hidden.clone()
        factor = 1 if prediction_step == 0 else 1 / math.sqrt(32)
        changed[:, -1, :] = hidden[:, -1, :] + (factor * delta).to(hidden.dtype)
        prediction_step += 1
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = runtime.layer.register_forward_hook(hook) if delta is not None else None
    try:
        with torch.no_grad():
            for _ in range(32):
                output = runtime.forward(
                    ids, mask, past_key_values=past, use_cache=True, logits_to_keep=1
                )
                scores = output.logits[:, -1, :].float()
                if not bool(torch.isfinite(scores).all()):
                    raise ValueError("Nonfinite training-generation logits.")
                scores[:, runtime.eos_ids] = -torch.inf
                ids = scores.argmax(-1)[:, None]
                tokens.append(int(ids[0, 0]))
                past = output.past_key_values
                mask = torch.cat([mask, torch.ones_like(ids)], -1)
    finally:
        if handle is not None:
            handle.remove()
    return tokens


def sequence_ie(runtime, torch, ids, start, endpoint, answer_label_ids=None):
    with torch.no_grad():
        base = logits(runtime, torch, ids, start, None, answer_label_ids)
        steer = logits(runtime, torch, ids, start, endpoint, answer_label_ids)
        if answer_label_ids is None:
            base[:, runtime.eos_ids] = -torch.inf
            steer[:, runtime.eos_ids] = -torch.inf
        old, new = (base.argmax(-1), steer.argmax(-1))
        keep = old != new
        indices = torch.nonzero(keep).flatten()
        count = int(keep.sum())
    total = torch.zeros_like(endpoint, dtype=torch.float64)
    if not count:
        return total, 0
    # Integrate only positions whose preferred token changes between endpoints.
    # Midpoints avoid placing a gradient sample directly on either endpoint.
    for step in range(P["attribution"]["midpoints"]):
        fraction = (step + 0.5) / P["attribution"]["midpoints"]
        delta = (endpoint * fraction).detach().requires_grad_(True)
        scores = logits(runtime, torch, ids, start, delta, answer_label_ids)
        metric = (scores[indices, new[indices]] - scores[indices, old[indices]]).sum()
        (grad,) = torch.autograd.grad(metric, delta)
        if not bool(torch.isfinite(grad).all()):
            raise ValueError("Nonfinite integrated gradient.")
        total += grad.detach().double() / P["attribution"]["midpoints"]
    return total, count


def compute(runtime, torch, rows, direction, scale, kind, folder, *, allow_empty=False):
    from sparse_steering.tasks.answer_selection import binary_prompt, generation_prompt

    target = folder / "attribution.json"
    freeze(
        folder / "attribution_identity.json",
        dict(
            train=digest(rows),
            direction=digest(direction),
            scale=scale,
            kind=kind,
            settings_sha256=sha(SETTINGS_PATH),
        ),
    )
    if target.exists():
        if read(folder / "attribution_COMPLETE.json")["sha256"] != sha(target):
            raise ValueError("Frozen attribution changed.")
        return read(target)
    endpoint = torch.tensor(
        direction, dtype=torch.float32, device=next(runtime.model.parameters()).device
    ) * (math.sqrt(P["attribution"]["edit_norm_ratio_squared"]) * scale)
    partial = folder / "attribution_partial.json"
    state = (
        read(partial)
        if partial.exists()
        else dict(next_question=0, total=[0.0] * len(direction), positions=0, sequences=0)
    )
    total = torch.tensor(
        state["total"], dtype=torch.float64, device=next(runtime.model.parameters()).device
    )
    versions = [(id(p), p._version) for p in runtime.model.parameters()]
    for index in range(state["next_question"], len(rows)):
        row = rows[index]
        if kind == "qa":
            sequences = []
            for order in (0, 1):
                prefix = runtime.tokenizer.encode(binary_prompt(row, order))
                sequences.append((prefix, len(prefix) - 1, runtime.answer_label_ids))
        else:
            prefix = runtime.tokenizer.encode(generation_prompt(row))
            if len(prefix) + 32 > runtime.model.config.max_position_embeddings:
                raise ValueError("Training generation exceeds context.")
            path = folder / "trajectories" / f"{index:04d}.json"
            if not path.exists():
                freeze(
                    path,
                    dict(
                        id=row["id"],
                        prefix=prefix,
                        base=generation(runtime, torch, prefix, None),
                        steered=generation(runtime, torch, prefix, endpoint),
                    ),
                )
            saved = read(path)
            if saved["id"] != row["id"] or saved["prefix"] != prefix:
                raise ValueError("Training trajectory identity differs.")
            sequences = [
                (prefix + saved[mode][:-1], len(prefix) - 1, None) for mode in ("base", "steered")
            ]
        for ids, start, answer_label_ids in sequences:
            grad, count = sequence_ie(runtime, torch, ids, start, endpoint, answer_label_ids)
            total += grad
            state["positions"] += count
            state["sequences"] += 1
        if versions != [(id(p), p._version) for p in runtime.model.parameters()] or any(
            (p.requires_grad or p.grad is not None for p in runtime.model.parameters())
        ):
            raise ValueError("Attribution changed model weights.")
        state.update(next_question=index + 1, total=total.cpu().tolist())
        if (index + 1) % 8 == 0 or index + 1 == len(rows):
            save(partial, state)
            print(
                json.dumps(
                    dict(
                        stage="attribution",
                        questions=index + 1,
                        positions=state["positions"],
                        time=time.time(),
                    )
                ),
                flush=True,
            )
    if state["positions"] == 0 and (not allow_empty):
        raise ValueError(
            "The steered and unsteered predictions never differ at the attribution strength."
        )
    mean = total / max(1, state["positions"])
    effect = endpoint.double() * mean
    payload = dict(
        signed_gradient=mean.cpu().tolist(),
        indirect_effect=effect.cpu().tolist(),
        questions=len(rows),
        positions=state["positions"],
        sequences=state["sequences"],
        endpoint=endpoint.cpu().tolist(),
        settings_sha256=sha(SETTINGS_PATH),
        model_parameter_gradients=False,
    )
    freeze(target, payload)
    freeze(folder / "attribution_COMPLETE.json", dict(sha256=sha(target), completed=time.time()))
    return payload
