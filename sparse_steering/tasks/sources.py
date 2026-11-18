"""Download the pinned public task sources and verify source checksums."""

import urllib.request

from sparse_steering.runtime.storage import sha

TQA_SHA = "b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c"
HALU_SHA = "89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88"
HALU_URL = (
    "https://raw.githubusercontent.com/RUCAIBox/HaluEval/"
    "b7253db3cdaa0ab2c382f92b26b390109174f77e/data/qa_data.json"
)
PARA_REV = "958b799a4c3a7c16463632465aaaa8ad083235a8"
RTP_REV = "f21629712ffd6a3d13a54fd2807ccd521c55ef74"


def download(url, target, expected=None):
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=180) as response:
            payload = response.read()
        temp = target.with_suffix(".download")
        temp.write_bytes(payload)
        temp.replace(target)
    if expected and sha(target) != expected:
        raise ValueError("Frozen source hash differs: " + str(target))
    return target
