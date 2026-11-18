"""Read settings, protect output locations, and save scientific artifacts."""

import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def freeze(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Immutable record changed: {path}")
    else:
        save(path, value)


def progress(**values):
    print(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **values}), flush=True)


SETTINGS_PATH = CODE / "protocol/experiment.json"
SETTINGS = read(SETTINGS_PATH)


def metadata(start=None, units=None, gpu=True):
    return dict(
        library_versions={
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")
        }
    )


def path(variable, *parts):
    root = Path(
        os.environ.get(
            "LLM_TASK_ROOT",
            os.environ.get(
                "SPARSE_STEERING_RUN_ROOT", Path.home() / ".local/share/sparse_steering"
            ),
        )
    ).resolve()
    target = Path(os.environ[variable]).joinpath(*map(str, parts))
    if not target.resolve().is_relative_to(root):
        raise ValueError("Output escapes configured run root: " + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def guard(gpu=False):
    root = Path(
        os.environ.get(
            "LLM_TASK_ROOT",
            os.environ.get(
                "SPARSE_STEERING_RUN_ROOT", Path.home() / ".local/share/sparse_steering"
            ),
        )
    ).resolve()
    if root == CODE or root.is_relative_to(CODE):
        raise ValueError("Keep experiment state outside the release source folder.")
    if gpu:
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA GPU with native bf16 support is required.")
