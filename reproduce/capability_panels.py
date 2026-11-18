"""Prepare fixed capability panels from the pinned public datasets."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from sparse_steering.runtime.storage import CODE, guard, read, save, sha


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    guard()
    from datasets import load_dataset

    from sparse_steering.evaluation.panels import preprocess

    folder = Path(os.environ["DATA_DIR"]) / "capability_panels"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = read(CODE / "configs/dataset_revisions.json")
    ids = read(CODE / "configs/capability_item_ids.json")
    for name, spec in manifest["datasets"].items():
        data = load_dataset(
            spec["repository"],
            spec["subset"],
            split=spec["split"],
            revision=spec["revision"],
            cache_dir=os.environ["HF_DATASETS_CACHE"],
        )
        rows = []
        for item in ids[name]:
            index = item["source_index"]
            r = data[index]
            if name == "arc_challenge":
                choices = r["choices"]["text"]
                answer = r["choices"]["label"].index(r["answerKey"])
                key = r["id"]
                prompt = "Question: " + r["question"] + "\nAnswer:"
            else:
                choices = [preprocess(x) for x in r["endings"]]
                answer = int(r["label"])
                key = f"validation:{index}:ind:{r['ind']}"
                prompt = preprocess(
                    r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()
                )
            if str(key) != item["id"] or answer != item["answer"]:
                raise ValueError("Fixed capability item changed.")
            rows.append(
                dict(id=str(key), source_index=index, prompt=prompt, choices=choices, answer=answer)
            )
        save(folder / (name + ".json"), rows)
    data = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="test",
        revision="b08601e04326c79dfdd32d625aee71d232d685c3",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    )
    corpus = "\n".join((r["text"] for r in data if r["text"].strip()))
    size = 8192
    blocks = [corpus[i : i + size] for i in range(0, len(corpus) - size + 1, size)]

    def rank(i):
        return hashlib.sha256(
            json.dumps([20261015, "wikitext", int(i)], sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    indices = sorted(sorted(range(len(blocks)), key=rank)[:128])
    (folder / "wikitext.json").write_text(
        json.dumps(
            [dict(id=f"wikitext:{i}", source_index=i, text=blocks[i]) for i in indices],
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    expected = manifest["files_sha256"]
    for name in ("arc_challenge.json", "hellaswag.json"):
        if sha(folder / name) != expected[name]:
            raise ValueError("Fixed panel serialization differs: " + name)
    if (
        sha(folder / "wikitext.json")
        != "c19c75a5184b2cb11ecd8562fe446c508d6d1548a7db915ff4aacdb29de502a9"
    ):
        raise ValueError("WikiText fixed panel differs.")
    save(
        folder / "MANIFEST.json",
        dict(
            files_sha256={
                name: sha(folder / name)
                for name in ("arc_challenge.json", "hellaswag.json", "wikitext.json")
            }
        ),
    )


if __name__ == "__main__":
    main()
