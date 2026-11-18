"""Train coordinate scores using neutral and toxic full-response preferences."""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import torch

from sparse_steering.method.mask_training.checkpoints import atomic_torch, rankable_mask_logits
from sparse_steering.method.mask_training.masks import (
    exact_topk_mask,
    straight_through_exact_k_direction,
)
from sparse_steering.method.mask_training.objectives import preference_loss
from sparse_steering.runtime.storage import SETTINGS, digest, freeze, progress, read, save

dump = save


@torch.no_grad()
def references(runtime, panels, artifacts):
    path = artifacts / "reference_logps.json"
    identity = dict(
        model=runtime.spec,
        panels=digest(panels),
        response_reduction="sum",
        include_native_eos=True,
        prompt_loss_masked=True,
        intervention="unsteered",
    )
    freeze(artifacts / "reference_identity.json", identity)
    values = read(path) if path.exists() else {}
    for split, rows in panels.items():
        for index, row in enumerate(rows):
            if row["id"] in values:
                continue
            scores, counts = runtime.paired_logps([row])
            values[row["id"]] = dict(
                logps=scores[0].cpu().tolist(), tokens=counts[0].cpu().tolist()
            )
            if (index + 1) % 32 == 0:
                dump(path, values)
                progress(
                    stage="reference_logps",
                    model=runtime.model_key,
                    split=split,
                    questions=index + 1,
                )
    dump(path, values)
    if set(values) != {r["id"] for rows in panels.values() for r in rows}:
        raise ValueError("Reference cache has missing or foreign pairs.")
    return values


@torch.no_grad()
def evaluate_candidate(runtime, rows, direction, scale, reference, *, training_config=None):
    # Copy defaults only for direct calls; the runner passes case settings.
    if training_config is None:
        training_config = deepcopy(SETTINGS["mask_training"]["open_generation"])
    losses, improvements = ([], {1: [], -1: []})
    for edit_norm_ratio_squared in training_config["mask"]["fit_edit_norm_ratios_squared"]:
        for row in rows:
            ref = torch.tensor([reference[row["id"]]["logps"]], device="cuda")
            for sign in (1, -1):
                scores, counts = runtime.paired_logps(
                    [row], direction, math.sqrt(edit_norm_ratio_squared) * scale, sign
                )
                if counts[0].cpu().tolist() != reference[row["id"]]["tokens"]:
                    raise ValueError("Reference response token count changed.")
                losses.append(
                    float(
                        preference_loss(scores, ref, sign, training_config["mask"]["gamma"]).mean()
                    )
                )
                gap = scores[0, 0] - scores[0, 1] - (ref[0, 0] - ref[0, 1])
                improvements[sign].append(float(sign * gap))
    return dict(
        preference_loss=float(np.mean(losses)),
        positive_preference_treatment=float(np.mean(improvements[1])),
        negative_preference_treatment=float(np.mean(improvements[-1])),
    )


def fit_mask(
    runtime,
    train,
    selection,
    dense,
    scale,
    artifacts,
    smoke=False,
    *,
    step_callback=None,
    training_config=None,
):
    # Copy defaults only for direct calls; the runner passes case settings.
    if training_config is None:
        training_config = deepcopy(SETTINGS["mask_training"]["open_generation"])
    output = artifacts / "selected_mask.json"
    if output.exists():
        return np.asarray(read(output)["mask"], dtype=bool)
    settings = training_config["mask"]
    k = math.ceil(len(dense) * settings["update_fraction"])
    dense_tensor = torch.tensor(dense, dtype=torch.float32, device="cuda")
    reference = references(runtime, dict(train=train, mask_selection=selection), artifacts)
    candidate_path = artifacts / "mask_candidates.json"
    candidates = read(candidate_path) if candidate_path.exists() else []
    epochs = 1 if smoke else settings["epochs"]
    for rerun in [0] if smoke else settings["reruns"]:
        checkpoint = artifacts / f"mask_rerun{rerun}.pt"
        torch.manual_seed(training_config["seed"] + rerun)
        logits = torch.nn.Parameter(torch.randn(len(dense), device="cuda") * 0.1)
        optimizer = torch.optim.AdamW([logits], lr=settings["learning_rate"], weight_decay=0.0)
        nsteps = math.ceil(len(train) / settings["question_batch"])
        step = 0
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
            with torch.no_grad():
                logits.copy_(saved["logits"])
            optimizer.load_state_dict(saved["optimizer"])
            step = saved["step"]
        for epoch in range(epochs):
            if any((c["rerun"] == rerun and c["epoch"] == epoch + 1 for c in candidates)):
                continue
            order = np.random.default_rng(training_config["seed"] + rerun * 10 + epoch).permutation(
                len(train)
            )
            for local_step, start in enumerate(range(0, len(train), settings["question_batch"])):
                absolute = epoch * nsteps + local_step
                if absolute < step:
                    continue
                batch = [train[int(i)] for i in order[start : start + settings["question_batch"]]]
                temperature = 2 * (0.25 / 2) ** (absolute / max(1, epochs * nsteps - 1))
                edit_norm_ratio_squared = settings["fit_edit_norm_ratios_squared"][
                    absolute % len(settings["fit_edit_norm_ratios_squared"])
                ]
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for row in batch:
                    ref = torch.tensor([reference[row["id"]]["logps"]], device="cuda")
                    for sign in (1, -1):
                        effective = rankable_mask_logits(logits, dense_tensor, k)
                        direction = straight_through_exact_k_direction(
                            effective, dense_tensor, k=k, temperature=temperature
                        ).direction
                        scores, counts = runtime.paired_logps(
                            [row], direction, math.sqrt(edit_norm_ratio_squared) * scale, sign
                        )
                        if counts[0].cpu().tolist() != reference[row["id"]]["tokens"]:
                            raise ValueError("Reference response token count changed.")
                        loss = preference_loss(scores, ref, sign, settings["gamma"]).mean()
                        (loss / (2 * len(batch))).backward()
                        total_loss += float(loss.detach()) / (2 * len(batch))
                if (
                    logits.grad is None
                    or not bool(torch.isfinite(logits.grad).all())
                    or int(torch.count_nonzero(logits.grad)) == 0
                ):
                    raise ValueError("Missing, exactly zero or nonfinite mask gradient.")
                gradient_norm = float(logits.grad.double().norm())
                torch.nn.utils.clip_grad_norm_([logits], 1.0)
                optimizer.step()
                step = absolute + 1
                if step % 12 == 0 or step % nsteps == 0:
                    atomic_torch(
                        checkpoint,
                        dict(logits=logits.detach(), optimizer=optimizer.state_dict(), step=step),
                    )
                    progress(
                        stage="preference_mask_fit",
                        model=runtime.model_key,
                        rerun=rerun,
                        epoch=epoch + 1,
                        step=step,
                        loss=total_loss,
                        gradient_norm=gradient_norm,
                    )
                if step_callback is not None:
                    hard = exact_topk_mask(
                        rankable_mask_logits(logits.detach(), dense_tensor, k).cpu().numpy(), k
                    )
                    step_callback(rerun, epoch, step, hard)
            mask = exact_topk_mask(
                rankable_mask_logits(logits.detach(), dense_tensor, k).cpu().numpy(), k
            )
            masked = dense.astype(np.float64) * mask
            rho = float(np.linalg.norm(masked))
            direction = torch.tensor(masked / rho, device="cuda", dtype=torch.float32)
            candidate = dict(
                rerun=rerun,
                epoch=epoch + 1,
                rho=rho,
                mask=mask.astype(int).tolist(),
                **evaluate_candidate(
                    runtime, selection, direction, scale, reference, training_config=training_config
                ),
            )
            candidates.append(candidate)
            dump(candidate_path, candidates)
            progress(
                stage="preference_checkpoint",
                model=runtime.model_key,
                **{key: value for key, value in candidate.items() if key != "mask"},
            )
    winner = min(candidates, key=lambda r: (r["preference_loss"], r["rerun"], r["epoch"]))
    freeze(output, winner)
    return np.asarray(winner["mask"], dtype=bool)
