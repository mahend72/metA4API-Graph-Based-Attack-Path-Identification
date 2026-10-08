"""Read / write the on-disk synthetic dataset (``data/synthetic/<profile>/*.json``)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .generator import SyntheticDataset, dumps


def write_dataset(ds: SyntheticDataset, out_dir: str | Path) -> Path:
    """Write the six JSON files.  ``metadata.json`` additionally records the sha256 of the other five files so a
    regenerated dataset can be compared byte-for-byte."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(ds.metadata)
    meta.pop("content_sha256", None)
    hashes = {}
    for name, obj in ds.payloads().items():
        if name == "metadata.json":
            continue
        text = dumps(obj)
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        hashes[name] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    meta["content_sha256"] = hashes
    ds.metadata["content_sha256"] = hashes
    (out / "metadata.json").write_text(dumps(meta), encoding="utf-8", newline="\n")
    return out


def read_dataset(in_dir: str | Path) -> SyntheticDataset:
    d = Path(in_dir)
    load = lambda n: json.loads((d / f"{n}.json").read_text(encoding="utf-8"))
    return SyntheticDataset(load("nodes"), load("edges"), load("labels"), load("campaigns"),
                            load("attack_paths"), load("metadata"))
