"""Download the frozen model revisions into the configured run root."""

import argparse
import os

from sparse_steering.runtime.storage import CODE, guard, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("gemma", "qwen", "llama", "qwen7", "qwen32b"), required=True)
    args = p.parse_args()
    guard()
    from huggingface_hub import snapshot_download

    spec = read(CODE / "protocol/experiment.json")["models"][args.model]
    snapshot_download(spec["id"], revision=spec["revision"], cache_dir=os.environ["HF_HUB_CACHE"])
    snapshot_download(
        "s-nlp/roberta_toxicity_classifier",
        revision="048c25bb1e199b98802784f96325f4840f22145d",
        cache_dir=os.environ["HF_HUB_CACHE"],
    )


if __name__ == "__main__":
    main()
