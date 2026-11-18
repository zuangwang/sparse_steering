"""Train coordinate scores using counterbalanced answer-label preferences."""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import torch

from sparse_steering.method.mask_training.objectives import ordered_pairs, reference_loss
from sparse_steering.runtime.storage import SETTINGS, digest, freeze, read, save, sha


def fit_mask(
    runtime, train, selection, dense, scale, folder, *, step_callback=None, training_config=None
):
    # Each fit owns its seed and mask settings; no module state is mutated.
    if training_config is None:
        training_config = deepcopy(SETTINGS["mask_training"]["answer_selection"])
    from sparse_steering.evaluation.strength_sweep import config
    from sparse_steering.method.mask_training.checkpoints import atomic_torch, rankable_mask_logits
    from sparse_steering.method.mask_training.masks import (
        exact_topk_mask,
        straight_through_exact_k_direction,
    )
    from sparse_steering.tasks.answer_selection import binary_prompt

    folder.mkdir(parents=True, exist_ok=True)
    settings = training_config["mask"]
    k = math.ceil(len(dense) * settings["update_fraction"])
    identity = dict(
        protocol=digest(training_config),
        model=runtime.spec,
        train=digest(train),
        mask_selection=digest(selection),
        dense=digest(dense.tolist()),
        scale=scale,
    )
    freeze(folder / "mask_identity.json", identity)
    if (folder / "MASK_COMPLETE.json").exists():
        receipt = read(folder / "MASK_COMPLETE.json")
        for name, expected in receipt["files"].items():
            if sha(folder / name) != expected:
                raise ValueError("Completed mask artifact changed: " + name)
        return np.asarray(read(folder / "selected_mask.json")["mask"], dtype=bool)
    versions = [(id(p), p._version) for p in runtime.model.parameters()]
    vector = torch.tensor(dense, device="cuda", dtype=torch.float32)
    signs = torch.tensor([1.0, -1.0, 1.0, -1.0], device="cuda")
    texts = [[binary_prompt(row, order) for order in (0, 1) for _ in (1, -1)] for row in train]
    reference_path = folder / "reference_logps.json"
    if not reference_path.exists():
        reference = []
        with torch.no_grad():
            for prompts in texts:
                logits, _ = runtime.binary_logits(prompts, None)
                pairs = ordered_pairs(logits, runtime.answer_label_ids)
                if not torch.equal(pairs[:, 0], pairs[:, 1]):
                    raise ValueError("Unsteered reference depends on the sign label.")
                reference.append(pairs.cpu().tolist())
        freeze(reference_path, dict(identity=digest(identity), values=reference))
    ref = read(reference_path)
    if ref["identity"] != digest(identity) or len(ref["values"]) != len(train):
        raise ValueError("Reference cache differs from fitting inputs.")
    reference = torch.tensor(ref["values"], device="cuda", dtype=torch.float32)
    candidates_path = folder / "mask_candidates.json"
    candidates = read(candidates_path) if candidates_path.exists() else []
    nsteps = math.ceil(len(train) / settings["question_batch"])
    for rerun in settings["reruns"]:
        checkpoint = folder / f"mask_rerun{rerun}.pt"
        torch.manual_seed(training_config["seed"] + rerun)
        scores = torch.nn.Parameter(torch.randn(len(dense), device="cuda") * 0.1)
        optimizer = torch.optim.AdamW([scores], lr=settings["learning_rate"], weight_decay=0.0)
        step = 0
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
            if saved["identity"] != digest(identity):
                raise ValueError("Checkpoint belongs to a different objective or input.")
            with torch.no_grad():
                scores.copy_(saved["scores"])
            optimizer.load_state_dict(saved["optimizer"])
            step = saved["step"]
        for epoch in range(settings["epochs"]):
            if any((c["rerun"] == rerun and c["epoch"] == epoch + 1 for c in candidates)):
                if step < (epoch + 1) * nsteps:
                    raise ValueError("Candidate has no corresponding optimizer checkpoint.")
                continue
            order = np.random.default_rng(training_config["seed"] + rerun * 10 + epoch).permutation(
                len(train)
            )
            for local_step, start in enumerate(range(0, len(train), settings["question_batch"])):
                absolute = epoch * nsteps + local_step
                if absolute < step:
                    continue
                indices = order[start : start + settings["question_batch"]]
                temperature = 2 * (0.25 / 2) ** (absolute / max(1, settings["epochs"] * nsteps - 1))
                edit_norm_ratio_squared = settings["fit_edit_norm_ratios_squared"][
                    absolute % len(settings["fit_edit_norm_ratios_squared"])
                ]
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                gains = np.zeros(2)
                for index in indices:
                    direction = straight_through_exact_k_direction(
                        rankable_mask_logits(scores, vector, k),
                        vector,
                        k=k,
                        temperature=temperature,
                    ).direction
                    logits, _ = runtime.binary_logits(
                        texts[index],
                        direction,
                        math.sqrt(edit_norm_ratio_squared) * scale,
                        matched=False,
                        signs=signs,
                    )
                    loss, gain = reference_loss(
                        ordered_pairs(logits, runtime.answer_label_ids),
                        reference[index],
                        training_config["gamma"],
                    )
                    (loss / len(indices)).backward()
                    total_loss += float(loss.detach()) / len(indices)
                    gains += gain.mean(0).cpu().numpy() / len(indices)
                if (
                    scores.grad is None
                    or not bool(torch.isfinite(scores.grad).all())
                    or float(scores.grad.norm()) == 0
                ):
                    raise ValueError("Missing, zero or nonfinite score gradient.")
                gradient_norm = float(torch.nn.utils.clip_grad_norm_([scores], 1.0))
                optimizer.step()
                step = absolute + 1
                if any(
                    (p.requires_grad or p.grad is not None for p in runtime.model.parameters())
                ) or versions != [(id(p), p._version) for p in runtime.model.parameters()]:
                    raise ValueError("Frozen model changed or acquired gradients.")
                if step % 12 == 0 or step % nsteps == 0 or step == 1:
                    atomic_torch(
                        checkpoint,
                        dict(
                            identity=digest(identity),
                            scores=scores.detach(),
                            optimizer=optimizer.state_dict(),
                            step=step,
                        ),
                    )
                    record = dict(
                        stage="mask_fit",
                        rerun=rerun,
                        epoch=epoch + 1,
                        step=step,
                        loss=total_loss,
                        signed_reference_gains=gains.tolist(),
                        gradient_norm=gradient_norm,
                        temperature=temperature,
                        edit_norm_ratio_squared=edit_norm_ratio_squared,
                    )
                    save(folder / f"training/rerun{rerun}_step{step:03d}.json", record)
                    print(__import__("json").dumps(record), flush=True)
                if step_callback is not None:
                    hard = exact_topk_mask(
                        rankable_mask_logits(scores.detach(), vector, k).cpu().numpy(), k
                    )
                    step_callback(rerun, epoch, step, hard)
            mask = exact_topk_mask(
                rankable_mask_logits(scores.detach(), vector, k).cpu().numpy(), k
            )
            masked = dense.astype(np.float64) * mask
            rho = float(np.linalg.norm(masked))
            direction = torch.tensor(masked / rho, device="cuda", dtype=torch.float64)
            evaluations = []
            for edit_norm_ratio_squared in settings["fit_edit_norm_ratios_squared"]:
                runtime.configure(config("dense", edit_norm_ratio_squared), scale)
                evaluations.extend(runtime.evaluate_binary(selection, direction))
                runtime.prediction_trace = []
            candidate = dict(
                rerun=rerun,
                epoch=epoch + 1,
                step=step,
                probability=float(np.mean([r["correct_probability"] for r in evaluations])),
                nll=float(np.mean([r["correct_answer_label_nll"] for r in evaluations])),
                rho=rho,
                mask=mask.astype(int).tolist(),
                scores=scores.detach().cpu().tolist(),
            )
            candidates.append(candidate)
            save(candidates_path, candidates)
            print(
                __import__("json").dumps(
                    {
                        key: value
                        for key, value in candidate.items()
                        if key not in ("mask", "scores")
                    }
                ),
                flush=True,
            )
    if len(candidates) != len(settings["reruns"]) * settings["epochs"]:
        raise ValueError("Incomplete mask checkpoint comparison.")
    winner = min(candidates, key=lambda c: (-c["probability"], c["nll"], c["rerun"], c["epoch"]))
    freeze(folder / "selected_mask.json", winner)
    freeze(
        folder / "MASK_COMPLETE.json",
        dict(
            passed=True,
            steps_per_rerun=nsteps * settings["epochs"],
            model_frozen=True,
            reference_detached=True,
            files={
                p.name: sha(p)
                for p in folder.iterdir()
                if p.is_file() and p.suffix in (".json", ".pt")
            },
        ),
    )
    return np.asarray(winner["mask"], dtype=bool)
