"""Compute answer-selection and full-response preference losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def ordered_pairs(logits, answer_label_ids):
    if logits.ndim != 2 or logits.shape[0] != 4 or len(set(answer_label_ids)) != 2:
        raise ValueError("Four counterbalanced signed rows and two answer tokens required.")
    logps = logits.log_softmax(-1)[:, answer_label_ids]
    correct = torch.tensor([0, 0, 1, 1], device=logits.device)
    rows = torch.arange(4, device=logits.device)
    return torch.stack((logps[rows, correct], logps[rows, 1 - correct]), -1).reshape(2, 2, 2)


def reference_loss(policy, reference, gamma=5.0):
    if policy.shape != (2, 2, 2) or reference.shape != policy.shape:
        raise ValueError("Expected paired labels for both signs and A/B orders.")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("Positive finite loss scale required.")
    if not bool(torch.isfinite(policy).all() and torch.isfinite(reference).all()):
        raise ValueError("Nonfinite log probability.")
    reference = reference.detach()
    gain = policy[..., 0] - policy[..., 1] - (reference[..., 0] - reference[..., 1])
    signs = policy.new_tensor([1.0, -1.0])
    losses = F.softplus(-gamma * signs * gain)
    return losses.mean(), (signs * gain).detach()


def preference_loss(policy, reference, signs, gamma=0.1):
    if policy.shape != reference.shape or policy.shape[-1] != 2:
        raise ValueError("Matching paired response log probabilities required.")
    if not bool(torch.isfinite(policy).all() and torch.isfinite(reference).all()):
        raise ValueError("Nonfinite response log probability.")
    signs = torch.as_tensor(signs, dtype=policy.dtype, device=policy.device)
    if not bool(((signs == 1) | (signs == -1)).all()) or gamma <= 0:
        raise ValueError("The preference objective requires signs +/-1 and positive gamma.")
    # Paper preference-loss equation: gamma scales the reference-relative gap.
    gap = policy[..., 0] - policy[..., 1] - (reference[..., 0] - reference[..., 1])
    return F.softplus(-gamma * signs * gap)


def response_tokens(tokenizer, prompt, response, max_tokens):
    prefix = tokenizer.encode(prompt)
    complete = tokenizer.encode(prompt + " " + response)
    if complete[: len(prefix)] != prefix or len(complete) <= len(prefix):
        raise ValueError("Response must extend the exact prompt token sequence.")
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int) or eos < 0:
        raise ValueError("Full-response preference scoring requires a native EOS token.")
    if complete[-1] != eos:
        complete.append(eos)
    if len(complete) > max_tokens:
        raise ValueError("Full response including EOS exceeds context; truncation forbidden.")
    return complete, len(prefix)


def summed_response_logps(logits, ids, attention_mask, response_starts):
    if logits.shape[:2] != ids.shape or attention_mask.shape != ids.shape:
        raise ValueError("Aligned full-sequence logits, tokens and attention required.")
    # Score the complete response, including EOS, while excluding prompt and
    # padding positions from the summed log probability.
    positions = attention_mask.cumsum(-1) - 1
    starts = torch.as_tensor(response_starts, device=ids.device)
    target_mask = attention_mask[:, 1:].bool() & (positions[:, 1:] >= starts[:, None])
    if not bool(target_mask.any(-1).all()):
        raise ValueError("Every sequence requires scored response tokens.")
    labels = ids[:, 1:].masked_fill(~target_mask, -100)
    losses = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2), labels, reduction="none", ignore_index=-100
    )
    return -losses.sum(-1), target_mask.sum(-1)
