"""Single entry point for model code to obtain a dataset, whether SYNTHETIC or REAL.

    ds = load_dataset("synthetic", profile="dev")                       # data/synthetic/dev/
    ds = load_dataset("semi_synthetic", profile="dev")                  # data/semi_synthetic/dev/ (real CVEs)
    ds = load_dataset("real", tikg_path=..., labels_path=..., severity_path=..., paths_path=...)

Both return the same :class:`Dataset` (TIKG + LabelSpace + severity + reference paths + metadata), so experiment code
never branches on the source.  The two sources are kept strictly separate on disk (``data/synthetic/`` vs. whatever
real paths the caller supplies) and the real-data loaders in ``tikg.py`` / ``labels.py`` are reused untouched.

Safety: ``Dataset.is_synthetic`` is always available, and :meth:`Dataset.require_real` must be called by any code that
writes *reported* results - it raises for synthetic data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .labels import LabelSpace, ReferencePath, build_label_space, load_labels, load_reference_paths, load_severity
from .tikg import TIKG, load_tikg, tikg_from_dict

SYNTHETIC_ROOT = Path("data/synthetic")
SEMI_SYNTHETIC_ROOT = Path("data/semi_synthetic")


class SyntheticDataError(RuntimeError):
    pass


@dataclass
class Dataset:
    source: str                                   # "synthetic" | "semi_synthetic" | "real"
    name: str
    tikg: TIKG
    labels: LabelSpace
    severity: np.ndarray
    severity_known: np.ndarray
    reference_paths: List[ReferencePath]
    metadata: Dict[str, Any] = field(default_factory=dict)
    campaigns: Optional[Dict[str, Any]] = None    # synthetic only: campaign / family descriptions

    @property
    def is_synthetic(self) -> bool:
        """True for fully synthetic AND semi-synthetic data: neither may back reported results."""
        return self.source in ("synthetic", "semi_synthetic") or \
            self.metadata.get("dataset_type") in ("synthetic", "semi_synthetic")

    @property
    def is_semi_synthetic(self) -> bool:
        return self.metadata.get("dataset_type") == "semi_synthetic"

    def require_real(self) -> "Dataset":
        if self.is_synthetic:
            raise SyntheticDataError(
                f"dataset {self.name!r} is {self.source.upper()} (development/pipeline validation only); "
                "its results must not be reported as experimental results")
        return self

    def banner(self) -> str:
        return (f"[{self.source.upper()}] {self.name}: {self.tikg.n} nodes, {len(self.tikg.triplets)} relations"
                + ("  -- NOT FOR REPORTED RESULTS" if self.is_synthetic else ""))


def _load_synthetic(profile: str, root: Path, kind: str = "synthetic") -> Dataset:
    d = root / profile
    if not d.is_dir():
        raise FileNotFoundError(f"{d} not found; run: python scripts/generate_synthetic_tikg.py --profile {profile}")
    rd = lambda n: json.loads((d / f"{n}.json").read_text(encoding="utf-8"))
    meta = rd("metadata")
    if not (meta.get("dataset_type") == kind and meta.get("not_for_reported_experimental_results") is True):
        raise SyntheticDataError(f"{d} is not marked as {kind}/not-for-results; refusing to load it as {kind}")
    nodes, edges = rd("nodes"), rd("edges")
    g = tikg_from_dict({"entities": nodes,
                        "triplets": [{"s": e["source"], "r": e["relation"], "t": e["target"]} for e in edges]})
    labels = load_labels(g, d / "labels.json")
    sev = {n["id"]: n["attrs"]["cvss_base_score"] for n in nodes if n["attrs"].get("cvss_base_score") is not None}
    severity = np.zeros(g.n, dtype=np.float32)
    known = np.zeros(g.n, dtype=bool)
    for k, v in sev.items():
        severity[g.index[k]] = float(v) / 10.0
        known[g.index[k]] = True
    return Dataset(kind, f"{kind}/{profile}", g, labels, severity, known,
                   load_reference_paths(d / "attack_paths.json"), meta, rd("campaigns"))


def _load_real(tikg_path, labels_path, severity_path, paths_path, name: str) -> Dataset:
    if tikg_path is None or labels_path is None:
        raise ValueError("real data needs at least tikg_path and labels_path")
    g = load_tikg(tikg_path)
    labels = load_labels(g, labels_path)
    if severity_path:
        severity, known = load_severity(g, severity_path)
    else:
        severity, known = np.zeros(g.n, dtype=np.float32), np.zeros(g.n, dtype=bool)
    refs = load_reference_paths(paths_path) if paths_path else []
    return Dataset("real", name, g, labels, severity, known, refs, {"dataset_type": "real"})


def load_dataset(source: str = "synthetic", *, profile: str = "dev", root: str | Path | None = None,
                 tikg_path=None, labels_path=None, severity_path=None, paths_path=None,
                 name: str = "real") -> Dataset:
    if source == "synthetic":
        return _load_synthetic(profile, Path(root or SYNTHETIC_ROOT))
    if source == "semi_synthetic":
        return _load_synthetic(profile, Path(root or SEMI_SYNTHETIC_ROOT), "semi_synthetic")
    if source == "real":
        return _load_real(tikg_path, labels_path, severity_path, paths_path, name)
    raise ValueError(f"unknown source {source!r} (use 'synthetic', 'semi_synthetic' or 'real')")
