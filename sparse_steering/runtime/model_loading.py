"""Load the pinned model and provide shared encoding, activation and generation methods."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer

from sparse_steering.evaluation.strength_sweep import config
from sparse_steering.method.mask_training.objectives import response_tokens, summed_response_logps
from sparse_steering.runtime.hooks import make_dense_hook
from sparse_steering.runtime.storage import SETTINGS
from sparse_steering.tasks.answer_selection import binary_prompt, generation_prompt, question_scores

CausalModel = getattr(transformers, "AutoModelFor" + "CausalLM")
P = dict(generation=SETTINGS["generation"], models=SETTINGS["models"])


class BaseRuntime:

    def forward(self, ids, mask, **kwargs):
        # Left padding must not shift token positions, including cached decoding.
        positions = (mask.cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1] :]
        return self.model(input_ids=ids, attention_mask=mask, position_ids=positions, **kwargs)

    @torch.no_grad()
    def states(self, texts):
        collected = []

        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            collected.append(hidden[:, -1, :].float().cpu().numpy())

        handle = self.layer.register_forward_hook(capture)
        try:
            for start in range(0, len(texts), self.batch_size):
                encoded = self.encode(texts[start : start + self.batch_size])
                self.forward(
                    encoded.input_ids, encoded.attention_mask, use_cache=False, logits_to_keep=1
                )
        finally:
            handle.remove()
        return np.concatenate(collected)

    def binary_logits(self, texts, direction=None, target_norm=0, *, matched=True, signs=None):
        encoded = self.encode(texts)
        handle, records = (None, [])
        if direction is not None:
            hook, records = self.make_hook(direction, target_norm, matched=matched, signs=signs)
            handle = self.layer.register_forward_hook(hook)
        try:
            logits = (
                self.forward(
                    encoded.input_ids, encoded.attention_mask, use_cache=False, logits_to_keep=1
                )
                .logits[:, -1, :]
                .float()
            )
        finally:
            if handle is not None:
                handle.remove()
        return logits, records


class DenseMixin:

    def configure(self, config, scale):
        self.cell, self.scale, self.prediction_trace = (config, scale, [])

    def aggregate(self, records, count, prediction_steps):
        if not records:
            if self.cell["method"] != "unsteered":
                raise ValueError("Missing intervention.")
            return [
                dict(
                    realized_squared_edit_norm=0.0,
                    requested_squared_edit_norm=0.0,
                    prediction_step_count=0,
                )
                for _ in range(count)
            ]
        if len(records) != prediction_steps:
            raise ValueError(
                "The hook count differs from the expected number of prediction positions."
            )
        actual = (
            torch.stack([r["realized_squared_edit_norm"] for r in records]).sum(0).cpu().tolist()
        )
        request = (
            torch.stack([r["requested_squared_edit_norm"] for r in records]).sum(0).cpu().tolist()
        )
        return [
            dict(
                realized_squared_edit_norm=a,
                requested_squared_edit_norm=b,
                prediction_step_count=prediction_steps,
            )
            for a, b in zip(actual, request)
        ]

    @torch.no_grad()
    def evaluate_binary(self, rows, direction, target_norm=0):
        examples = [(r, o) for r in rows for o in (0, 1)]
        results = []
        for start in range(0, len(examples), self.batch_size):
            batch = examples[start : start + self.batch_size]
            self._contexts = [dict(id=r["id"], order=o) for r, o in batch]
            self._prediction_step = 0
            logits, records = self.binary_logits(
                [binary_prompt(r, o) for r, o in batch], direction, target_norm
            )
            scores = {
                k: v.cpu().numpy()
                for k, v in question_scores(
                    logits, self.answer_label_ids, [o for _, o in batch]
                ).items()
            }
            audits = self.aggregate(records, len(batch), 1)
            results.extend(
                (
                    dict(
                        id=r["id"],
                        order=o,
                        **{k: float(v[i]) for k, v in scores.items()},
                        **audits[i],
                    )
                    for i, (r, o) in enumerate(batch)
                )
            )
        return results


class Runtime(DenseMixin, BaseRuntime):

    def __init__(self, model, batch_size=None):
        self.model_key, self.spec = (model, P["models"][model])
        path = (
            Path(os.environ["HF_HUB_CACHE"])
            / ("models--" + self.spec["id"].replace("/", "--"))
            / "snapshots"
            / self.spec["revision"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            CausalModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                attn_implementation=self.spec["attention"],
                local_files_only=True,
            )
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        self.layer = self.model.model.layers[self.spec["layer"]]
        if self.model.config.hidden_size != self.spec["width"]:
            raise ValueError("Unexpected model hidden width.")
        self.batch_size = batch_size or P["generation"]["batch_size"]
        self.prompt_format = "raw"
        probe = binary_prompt(dict(question="What is 2 + 2?", correct="4", incorrect="5"), 0)
        prefix = self.tokenizer.encode(probe)
        self.answer_label_ids = []
        for label in ("A", "B"):
            full = self.tokenizer.encode(probe + " " + label)
            if full[:-1] != prefix:
                raise ValueError("Answer label must extend the actual prompt by exactly one token.")
            self.answer_label_ids.append(full[-1])
        if len(set(self.answer_label_ids)) != 2:
            raise ValueError("Answer labels collide.")
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = sorted(
            set(
                ([eos] if isinstance(eos, int) else list(eos or []))
                + ([self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else [])
            )
        )
        self.extra_normalization_mask = None
        self.configure(config("unsteered", 0), 1.0)

    def encode_ids(self, ids):
        if any((not x or len(x) > P["generation"]["max_input_tokens"] for x in ids)):
            raise ValueError("Empty/overlong input: truncation forbidden.")
        return self.tokenizer.pad(dict(input_ids=ids), padding=True, return_tensors="pt").to("cuda")

    def encode(self, texts):
        return self.encode_ids([self.tokenizer.encode(t) for t in texts])

    @torch.no_grad()
    def generate(self, rows, direction, target_norm=0):
        self._contexts = [dict(id=r["id"]) for r in rows]
        self._prediction_step = 0
        encoded = self.encode([generation_prompt(r) for r in rows])
        ids, mask = (encoded.input_ids, encoded.attention_mask)
        if ids.shape[1] + 32 > self.model.config.max_position_embeddings:
            raise ValueError("Generation would exceed native model context.")
        prompt_ids, prompt_mask = (ids, mask)
        past, tokens, records, handle = (None, [], [], None)
        eos_seen = torch.zeros(len(rows), dtype=torch.bool, device="cuda")
        if direction is not None:
            hook, records = self.make_hook(direction)
            handle = self.layer.register_forward_hook(hook)
        try:
            for _ in range(32):
                output = self.forward(
                    ids, mask, past_key_values=past, use_cache=True, logits_to_keep=1
                )
                logits = output.logits[:, -1, :].float()
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Nonfinite generation logits.")
                for eos in self.eos_ids:
                    eos_seen |= logits.argmax(-1) == eos
                # Keep all completions at exactly 32 tokens for comparable scoring.
                logits[:, self.eos_ids] = -torch.inf
                ids = logits.argmax(-1)[:, None]
                tokens.append(ids)
                past = output.past_key_values
                mask = torch.cat([mask, torch.ones_like(ids)], -1)
        finally:
            if handle is not None:
                handle.remove()
        tokens = torch.cat(tokens, -1)
        del past, output
        # Rescore without a steering hook or its KV cache: completion NLL measures
        # likelihood under the unsteered model, not under the generation path.
        logits = (
            self.forward(
                torch.cat([prompt_ids, tokens], -1),
                torch.cat([prompt_mask, torch.ones_like(tokens)], -1),
                use_cache=False,
                logits_to_keep=33,
            )
            .logits[:, :-1, :]
            .float()
        )
        nll = (
            F.cross_entropy(logits.transpose(1, 2), tokens, reduction="none")
            .mean(-1)
            .cpu()
            .tolist()
        )
        audits = self.aggregate(records, len(rows), 32)
        result = []
        for i, (row, text) in enumerate(
            zip(rows, self.tokenizer.batch_decode(tokens, skip_special_tokens=True))
        ):
            grams = [tuple(text.split()[j : j + 4]) for j in range(max(0, len(text.split()) - 3))]
            result.append(
                dict(
                    id=row["id"],
                    text=text,
                    completion_nll=nll[i],
                    repeated_4gram=int(len(grams) != len(set(grams))),
                    empty=int(not text.strip()),
                    any_eos_preferred=int(eos_seen[i]),
                    **audits[i],
                )
            )
        return result

    def paired_logps(self, rows, direction=None, magnitude=0.0, sign=1):
        if sign not in (1, -1):
            raise ValueError("Invalid intervention sign.")
        sequences, starts = ([], [])
        for row in rows:
            for side in ("correct", "incorrect"):
                sequence, start = response_tokens(
                    self.tokenizer,
                    generation_prompt(row),
                    row[side],
                    min(
                        P["generation"]["max_input_tokens"],
                        self.model.config.max_position_embeddings,
                    ),
                )
                sequences.append(sequence)
                starts.append(start)
        encoded = self.encode_ids(sequences)
        handle = None
        if direction is not None:

            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                delta = (sign * magnitude * direction).to(hidden.dtype)
                # Preference training intervenes on every non-padding position.
                changed = hidden + encoded.attention_mask[:, :, None] * delta[None, None, :]
                return (changed, *output[1:]) if isinstance(output, tuple) else changed

            handle = self.layer.register_forward_hook(hook)
        try:
            output = self.forward(encoded.input_ids, encoded.attention_mask, use_cache=False)
            logps, counts = summed_response_logps(
                output.logits, encoded.input_ids, encoded.attention_mask, starts
            )
        finally:
            if handle is not None:
                handle.remove()
        return logps.reshape(-1, 2), counts.reshape(-1, 2)

    make_hook = make_dense_hook


def fit_direction(runtime, rows, artifacts):
    texts = [f"{generation_prompt(row)} {row['correct']}" for row in rows]
    texts += [f"{generation_prompt(row)} {row['incorrect']}" for row in rows]
    states = runtime.states(texts)
    # Preferred endpoints precede opposite endpoints; retain float64 means.
    positive, negative = np.split(states.astype(np.float64), 2)
    raw = positive.mean(0) - negative.mean(0)
    if np.linalg.norm(raw) < 1e-10:
        raise ValueError("Zero fitted direction.")
    dense = (raw / np.linalg.norm(raw)).astype(np.float32)
    scale = float(np.linalg.norm(states.astype(np.float64), axis=1).mean())
    np.savez_compressed(artifacts / "fit_states.npz", states=states, dense=dense, scale=scale)
    return dense, scale


def bind(spec):
    import transformers

    from sparse_steering.baselines.dense import QUANTIZATION_DIAGNOSTICS
    from sparse_steering.evaluation.sweep_checks import audit

    if torch.__version__ != "2.8.0+cu128" or transformers.__version__ != "4.55.0":
        raise ValueError("The model requires the pinned Torch and Transformers versions.")
    torch.set_num_threads(8)
    torch.manual_seed(SETTINGS["inference_seed"])
    runtime = Runtime(spec["model"], batch_size=4)
    return runtime, torch, np, audit, None, QUANTIZATION_DIAGNOSTICS
