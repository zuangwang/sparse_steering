"""Download pinned task sources and apply the fixed split memberships."""

import argparse
import csv
import hashlib
import io
import json
import os
import time

from sparse_steering.runtime.storage import CODE, guard, metadata, path, read, save, sha
from sparse_steering.tasks.sources import HALU_SHA, HALU_URL, PARA_REV, RTP_REV, TQA_SHA, download
from sparse_steering.tasks.splits import apply_frozen, build, ids_only, normalize


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    guard()
    started = time.time()
    target = path("OUTPUT_DIR", "preflight/PREPARED.json")
    if target.exists():
        return
    cache = path("LLM_EXPERIMENT_CACHE_ROOT", "sources")
    cache.mkdir(parents=True, exist_ok=True)
    truth = download(
        "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
        cache / "TruthfulQA.csv",
        TQA_SHA,
    )
    halu = download(HALU_URL, cache / "halu.jsonl", HALU_SHA)
    para = download(
        f"https://huggingface.co/datasets/s-nlp/paradetox/resolve/{PARA_REV}/train.tsv",
        cache / "train.tsv",
    )
    pools = dict(
        truthfulqa=[
            dict(
                id=f"truthfulqa:{i}",
                question=r["Question"].strip(),
                correct=r["Best Answer"].strip(),
                incorrect=r["Best Incorrect Answer"].strip(),
                answers=[r["Best Answer"].strip()],
            )
            for i, r in enumerate(
                csv.DictReader(io.StringIO(truth.read_text(encoding="utf-8-sig")))
            )
        ],
        halueval=[
            dict(
                id=f"halueval:{i}",
                question=r["question"],
                correct=r["right_answer"],
                incorrect=r["hallucinated_answer"],
                answers=[r["right_answer"]],
            )
            for i, r in enumerate(
                (__import__("json").loads(x) for x in halu.read_text().splitlines() if x.strip())
            )
        ],
        paradetox=[],
    )
    for i, r in enumerate(csv.DictReader(io.StringIO(para.read_text()), delimiter="\t")):
        toxic, neutral = (r["en_toxic_comment"].strip(), r["en_neutral_comment"].strip())
        if min(len(toxic.split()), len(neutral.split())) < 8 or normalize(toxic) == normalize(
            neutral
        ):
            continue
        pools["paradetox"].append(
            dict(
                id=f"paradetox:{i}",
                question="Rewrite this text into non-toxic language while keeping its meaning:\n"
                + toxic,
                correct=neutral,
                incorrect=toxic,
                answers=[neutral],
                binary_instruction=(
                    "Choose the non-toxic rewrite that preserves the original meaning."
                ),
            )
        )
    from datasets import load_dataset

    rtp = load_dataset(
        "allenai/real-toxicity-prompts",
        revision=RTP_REV,
        split="train",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    )
    pools["rtp"] = [
        dict(id=f"rtp:{i}", question="", prompt=r["prompt"]["text"], challenging=r["challenging"])
        for i, r in enumerate(rtp)
    ]
    splits = {}
    for seed in (0, 1, 2, 3):
        for task, rows in pools.items():
            source = CODE / f"protocol/splits/seed{seed}/{task}.json"
            manifest = read(source)
            if "sets" not in manifest:
                if task != "rtp" or seed != 0:
                    raise ValueError("Expected an explicit frozen membership file.")
                membership = ids_only(build(rows, task, seed))
                full = dict(
                    schema=1, seed=seed, task=task, sets=membership, sources=manifest["sources"]
                )
                encoded = (
                    json.dumps(full, indent=2, sort_keys=True, allow_nan=False) + "\n"
                ).encode()
                if hashlib.sha256(encoded).hexdigest() != manifest["expected_manifest_sha256"]:
                    raise ValueError("Reconstructed frozen RTP membership differs.")
                manifest = full
                save(
                    path("LLM_ARTIFACT_ROOT", "frozen_splits/seed0/rtp.json"),
                    dict(schema=1, seed=seed, task=task, sets=membership),
                )
            panels = apply_frozen(rows, manifest)
            if task == "rtp":
                panels = {
                    k: [{x: y for x, y in row.items() if x != "challenging"} for row in group]
                    for k, group in panels.items()
                }
            save(path("LLM_EXPERIMENT_CACHE_ROOT", f"panels/seed{seed}/{task}.json"), panels)
            splits[f"seed{seed}/{task}"] = dict(
                sha256=sha(source), counts={k: len(v) for k, v in panels.items()}
            )
    save(
        target,
        dict(
            status="prepared",
            splits=splits,
            sources={str(p): sha(p) for p in (truth, halu, para)},
            metadata=metadata(started, ["source preparation"], gpu=False),
        ),
    )
    print("Prepared the fixed panels for all four tasks and four seeds.", flush=True)


if __name__ == "__main__":
    main()
