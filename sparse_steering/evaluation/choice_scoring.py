"""Score ARC-Challenge and HellaSwag choices using cached teacher forcing."""


def encoded_panel(runtime, rows):
    flat = []
    for row in rows:
        context = row["prompt"]
        prompt_ids = runtime.tokenizer.encode(context, add_special_tokens=True)
        for choice, text in enumerate(row["choices"]):
            full = runtime.tokenizer.encode(context + " " + text, add_special_tokens=True)
            if full[: len(prompt_ids)] != prompt_ids or len(full) == len(prompt_ids):
                raise ValueError("Choice must extend the exact tokenized context: " + row["id"])
            if len(full) > runtime.model.config.max_position_embeddings:
                raise ValueError("Native context exceeded; no truncation is permitted.")
            flat.append(
                dict(
                    key=f"{row['id']}:{choice}",
                    id=row["id"],
                    choice=choice,
                    input_ids=full[:-1],
                    context_tokens=len(prompt_ids),
                    targets=full[len(prompt_ids) :],
                    choice_characters=len(text),
                )
            )
    return sorted(flat, key=lambda x: (len(x["input_ids"]), x["key"]))


def batches(flat, max_sequences=32, max_batch_tokens=4096):
    batch = []
    for row in flat:
        if batch and (
            len(batch) == max_sequences
            or (len(batch) + 1) * len(row["input_ids"]) > max_batch_tokens
        ):
            yield batch
            batch = []
        batch.append(row)
    if batch:
        yield batch


def summarize_trace(trace):
    if not trace:
        return dict(
            prediction_step_count=0,
            realized_squared_edit_norm=0.0,
            requested_squared_edit_norm=0.0,
            max_norm_relative_drift=0.0,
        )
    return dict(
        prediction_step_count=len(trace),
        realized_squared_edit_norm=sum((a["realized_squared_edit_norm"] for a in trace)),
        requested_squared_edit_norm=sum((a["requested_squared_edit_norm"] for a in trace)),
        max_norm_relative_drift=max((a["norm_relative_drift"] for a in trace)),
    )


def score_batch(runtime, batch):
    torch, original = (runtime.torch, runtime.original)
    with torch.inference_mode():
        contexts = [r["input_ids"][: r["context_tokens"]] for r in batch]
        encoded = runtime.tokenizer.pad(
            dict(input_ids=contexts), padding=True, return_tensors="pt"
        ).to("cuda")
        ids, mask, past = (encoded.input_ids, encoded.attention_mask, None)
        lengths = [len(r["targets"]) for r in batch]
        totals = torch.zeros(len(batch), dtype=torch.float64, device="cuda")
        original.prediction_trace = []
        original._prediction_step = 0
        active = torch.ones(len(batch), dtype=torch.bool, device="cuda")
        step = 0
        handle = None
        if runtime.direction is not None:
            inner, records = original.make_hook(runtime.direction)

            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                indices = torch.nonzero(active).flatten()
                original._contexts = [
                    dict(id=batch[i]["key"], prediction_step=step) for i in indices.cpu().tolist()
                ]
                original._prediction_step = step
                subset = inner(module, inputs, hidden.index_select(0, indices))
                changed = hidden.clone()
                changed.index_copy_(0, indices, subset)
                records.clear()
                return (changed, *output[1:]) if isinstance(output, tuple) else changed

            handle = original.layer.register_forward_hook(hook)
        try:
            for step in range(max(lengths)):
                # Teacher-force the complete choice, but intervene only at its
                # first 32 prediction positions, including the prompt endpoint.
                if step == 32 and handle is not None:
                    handle.remove()
                    handle = None
                active = torch.tensor([step < n for n in lengths], device="cuda")
                output = runtime.forward(
                    ids, mask, past_key_values=past, use_cache=True, logits_to_keep=1
                )
                logits = output.logits[:, -1, :].float()
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Nonfinite cached logits.")
                labels = torch.tensor(
                    [
                        (
                            r["targets"][step]
                            if step < len(r["targets"])
                            else runtime.tokenizer.pad_token_id
                        )
                        for r in batch
                    ],
                    device="cuda",
                )
                totals -= (
                    torch.nn.functional.cross_entropy(logits, labels, reduction="none").double()
                    * active
                )
                ids = labels[:, None]
                mask = torch.cat((mask, active.long()[:, None]), dim=1)
                past = output.past_key_values
        finally:
            if handle is not None:
                handle.remove()
        result = []
        for i, row in enumerate(batch):
            trace = [a for a in original.prediction_trace if a["id"] == row["key"]]
            expected = 0 if runtime.direction is None else min(32, len(row["targets"]))
            if len(trace) != expected:
                raise ValueError("Cached steering position count differs.")
            result.append(
                dict(
                    key=row["key"],
                    id=row["id"],
                    choice=row["choice"],
                    logp=float(totals[i]),
                    tokens=len(row["targets"]),
                    characters=row["choice_characters"],
                    **summarize_trace(trace),
                )
            )
        original.prediction_trace = []
        return result


def aggregate_choices(rows, choices):
    import numpy as np

    lookup = {(r["id"], r["choice"]): r for r in choices}
    result = []
    for row in rows:
        values = [lookup[row["id"], i] for i in range(len(row["choices"]))]
        logps = [v["logp"] for v in values]
        chars = [v["characters"] for v in values]
        tokens = [v["tokens"] for v in values]
        norm = [a / b for a, b in zip(logps, chars)]
        token_norm = [a / b for a, b in zip(logps, tokens)]
        result.append(
            dict(
                id=row["id"],
                source_index=row["source_index"],
                answer=row["answer"],
                choice_logps=logps,
                choice_characters=chars,
                choice_tokens=tokens,
                predicted=int(np.argmax(norm)),
                accuracy=float(np.argmax(norm) == row["answer"]),
                token_norm_accuracy=float(np.argmax(token_norm) == row["answer"]),
                unnormalized_accuracy=float(np.argmax(logps) == row["answer"]),
                tied_choices=int(sum((x == max(norm) for x in norm))),
                prediction_steps=sum((v["prediction_step_count"] for v in values)),
            )
        )
    return result
