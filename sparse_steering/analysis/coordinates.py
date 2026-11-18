"""Compute coordinate overlap and mask stability from fitted masks."""

import argparse
import gzip
import json
import statistics
from itertools import combinations

from sparse_steering.runtime.runner import loc
from sparse_steering.runtime.storage import guard, path, read, save


def choose(candidates, qa):
    return min(
        candidates,
        key=(
            (lambda c: (-c["probability"], c["nll"], c["rerun"], c["epoch"]))
            if qa
            else lambda c: (c["preference_loss"], c["rerun"], c["epoch"])
        ),
    )


def case(seed, model, task):
    folder = loc("LLM_ARTIFACT_ROOT", seed, model, task)
    if not (folder / "fits.json").exists():
        return
    fits = read(folder / "fits.json")
    dense = fits["dense"]["dense"]
    mask = read(loc("OUTPUT_DIR", seed, model, task, "mask.json"))
    selected = set(mask["coordinates"])
    k = len(selected)
    effect = read(folder / "attribution/attribution.json")["indirect_effect"]
    top = lambda scores: set(sorted(range(len(scores)), key=lambda i: (-abs(scores[i]), i))[:k])
    # Compare equal-size supports; use coordinate index to resolve ties.
    comparisons = dict(
        top_k_coefficients=top(dense),
        top_k_ie=top(effect),
        gradient={i for i, v in enumerate(fits["gradient"]["mask"]) if v},
        ie={i for i, v in enumerate(fits["ie"]["mask"]) if v},
    )
    total = sum((v * v for v in dense))
    stats = dict(
        k=k,
        width=len(dense),
        overlaps={name: len(selected & coords) / k for name, coords in comparisons.items()},
        selected_squared_coefficient_mass=sum((dense[i] ** 2 for i in selected)) / total,
        top_k_squared_coefficient_mass=sum(
            (dense[i] ** 2 for i in comparisons["top_k_coefficients"])
        )
        / total,
    )
    save(loc("OUTPUT_DIR", seed, model, task, "coordinate_statistics.json"), stats)
    candidates = read(folder / "rho20/mask_candidates.json")
    reruns = {}
    finals = {}
    trajectories = []
    for rerun in (0, 1, 2):
        own = [c for c in candidates if c["rerun"] == rerun]
        winner = choose(own, task != "paradetox")
        reruns[rerun] = dict(
            selected_epoch=winner["epoch"],
            coordinates=[i for i, v in enumerate(winner["mask"]) if v],
        )
        final = next((c for c in own if c["epoch"] == 2))
        finals[rerun] = {i for i, v in enumerate(final["mask"]) if v}
        files = sorted((folder / "rho20/step_masks").glob(f"rerun{rerun}_step*.json.gz"))
        if len(files) != 96:
            raise ValueError("Expected every optimizer update in trajectory.")
        for file in files:
            with gzip.open(file, "rt") as stream:
                record = json.load(stream)
            coordinates = set(record["coordinates"])
            if len(coordinates) != k:
                raise ValueError("Step mask coordinate count differs.")
            trajectories.append(
                dict(
                    rerun=rerun,
                    step=record["step"],
                    epoch=record["epoch"],
                    overlap_final=len(coordinates & finals[rerun]) / k,
                )
            )
    pairs = []
    for a, b in combinations((0, 1, 2), 2):
        pairs.append(
            dict(
                rerun_a=a,
                rerun_b=b,
                overlap=len(set(reruns[a]["coordinates"]) & set(reruns[b]["coordinates"])) / k,
            )
        )
    save(
        loc("OUTPUT_DIR", seed, model, task, "stability.json"),
        dict(
            k=k,
            reruns=reruns,
            pairs=pairs,
            pairwise_mean=statistics.mean((p["overlap"] for p in pairs)),
            pairwise_range=[min((p["overlap"] for p in pairs)), max((p["overlap"] for p in pairs))],
            pair_definition=(
                "Masks selected within each rerun using the unchanged epoch checkpoint rule"
            ),
            trajectory_definition=(
                "Post-update hard mask intersection final-epoch mask / K, steps 1..96"
            ),
            trajectories=trajectories,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=("gemma", "qwen", "llama", "qwen7", "qwen32b"), required=True
    )
    args = parser.parse_args()
    guard()
    model = args.model
    for seed in (0, 1, 2, 3):
        for task in ("truthfulqa", "halueval", "paradetox"):
            case(seed, model, task)
    across = {}
    for task in ("truthfulqa", "halueval", "paradetox"):
        masks = {
            seed: set(read(loc("OUTPUT_DIR", seed, model, task, "mask.json"))["coordinates"])
            for seed in (1, 2, 3)
            if loc("OUTPUT_DIR", seed, model, task, "mask.json").exists()
        }
        across[task] = dict(
            available_seeds=sorted(masks),
            descriptive_only=True,
            pairs=[
                dict(seed_a=a, seed_b=b, overlap=len(masks[a] & masks[b]) / len(masks[a]))
                for a, b in combinations(masks, 2)
            ],
        )
    save(path("OUTPUT_DIR", f"stability_across_seeds_{model}.json"), across)
    print("coordinate overlap and mask stability saved for available fits:", model, flush=True)


if __name__ == "__main__":
    main()
