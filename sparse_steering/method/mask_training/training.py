"""Apply experiment seeds and record coordinate masks during training."""

import gzip
import json
from copy import deepcopy
from pathlib import Path

from sparse_steering.runtime.storage import SETTINGS
from sparse_steering.tasks.splits import seed_value


def training_settings(seed, task, reruns, fraction):
    """Return independent objective settings for this fit, without shared mutation."""
    objective = "answer_selection" if task in ("truthfulqa", "halueval") else "open_generation"
    config = deepcopy(SETTINGS["mask_training"][objective])
    config["seed"] = seed_value(seed, task, training=True)
    config["mask"]["reruns"] = list(reruns)
    config["mask"]["update_fraction"] = fraction
    return config


def mask_recorder(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    def record(rerun, epoch, step, mask):
        target = folder / f"rerun{rerun}_step{step:03d}.json.gz"
        value = dict(
            rerun=rerun, epoch=epoch + 1, step=step, coordinates=mask.nonzero()[0].tolist()
        )
        temporary = target.with_suffix(".tmp")
        with gzip.open(temporary, "wt") as stream:
            json.dump(value, stream, separators=(",", ":"))
        temporary.replace(target)

    return record


def fit_mask(
    runtime,
    panels,
    dense,
    scale,
    folder,
    *,
    seed,
    task,
    reruns=(0, 1, 2),
    fraction=0.2,
    stop_requested=None,
):
    if task in ("truthfulqa", "halueval"):
        from sparse_steering.method.mask_training import answer_selection as trainer
    elif task == "paradetox":
        from sparse_steering.method.mask_training import open_generation as trainer
    else:
        raise ValueError("RTP reuses the ParaDetox mask; it has no trainer.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    recorder = mask_recorder(folder / "step_masks")

    def observe(rerun, epoch, step, mask):
        recorder(rerun, epoch, step, mask)
        if stop_requested is not None and stop_requested() and (step % 12 == 0):
            raise SystemExit(75)

    training_config = training_settings(seed, task, reruns, fraction)
    return trainer.fit_mask(
        runtime,
        panels["train"],
        panels["mask_selection"],
        dense,
        scale,
        folder,
        step_callback=observe,
        training_config=training_config,
    )
