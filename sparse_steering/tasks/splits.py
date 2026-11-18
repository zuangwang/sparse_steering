"""Build deterministic, disjoint training, mask-selection and test sets."""

import hashlib
import unicodedata

REFERENCE_SPLIT_SEEDS = dict(
    truthfulqa=20261009, halueval=20261009, paradetox=20261012, rtp=20261031
)
REFERENCE_TRAIN_SEEDS = dict(truthfulqa=20261009, halueval=20261009, paradetox=20261012)
COUNTS = {
    "truthfulqa": dict(train=384, mask_selection=128, validation=64, test=214),
    "halueval": dict(train=384, mask_selection=128, validation=128, test=512),
    "paradetox": dict(train=384, mask_selection=128, validation=128, test=512),
    "rtp": dict(validation=128, test=512),
}


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def seed_value(seed, task, *, training=False):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    defaults = REFERENCE_TRAIN_SEEDS if training else REFERENCE_SPLIT_SEEDS
    return defaults[task] if seed == 0 else 1000 * seed if training else seed


def training_seeds(seed, task, rerun, epoch):
    if rerun not in (0, 1, 2) or epoch not in (0, 1):
        raise ValueError("Expected three reruns and two epochs.")
    base = seed_value(seed, task, training=True)
    return dict(initialization=base + rerun, data_order=base + 10 * rerun + epoch)


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _slice(rows, counts):
    if len(rows) < sum(counts.values()):
        raise ValueError("Insufficient distinct source items.")
    result, offset = ({}, 0)
    for name, count in counts.items():
        result[name] = rows[offset : offset + count]
        offset += count
    ids = [r["id"] for group in result.values() for r in group]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source IDs across partitions.")
    return result


def build(rows, task, seed, *, excluded=(), counts=None):
    salt = seed_value(seed, task)
    counts = COUNTS[task] if counts is None else counts
    rows = list(rows)
    if task in ("truthfulqa", "halueval"):
        unique = {}
        for row in rows:
            key = normalize(row["question"])
            if key and key not in excluded and (key not in unique) and row["correct"].strip():
                unique[key] = row
        ranked = [unique[q] for q in sorted(unique, key=lambda q: _hash(f"{salt}|{task}|{q}"))]
    elif task == "paradetox":
        ordered = sorted(rows, key=lambda r: _hash(f"{salt}|{normalize(r['question'])}"))
        ranked, seen = ([], set())
        for row in ordered:
            keys = {normalize(row[k]) for k in ("question", "correct", "incorrect")}
            if "" in keys or keys & seen:
                continue
            seen.update(keys)
            ranked.append(row)
            if len(ranked) == sum(counts.values()):
                break
    elif task == "rtp":
        unique, seen = ([], set())
        for row in rows:
            key = normalize(row["prompt"])
            if not row["challenging"] or not key or key in seen:
                continue
            seen.add(key)
            unique.append(row)
        ranked = sorted(unique, key=lambda r: _hash(f"{salt}|rtp|{int(r['id'].split(':')[1])}"))
    else:
        raise ValueError("Unknown task: " + task)
    return _slice(ranked, counts)


def ids_only(panels):
    return {name: [row["id"] for row in rows] for name, rows in panels.items()}


def apply_frozen(rows, manifest):
    lookup = {r["id"]: r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Source IDs are not unique.")
    panels = manifest["sets"]
    ids = [i for group in panels.values() for i in group]
    if len(ids) != len(set(ids)):
        raise ValueError("Frozen membership overlaps.")
    return {name: [lookup[i] for i in group] for name, group in panels.items()}
