"""Label space, CVSS severity and reference-path loaders.

NO labels, CVSS scores or reference paths are shipped or inferred.  These loaders consume user-supplied files
(formats in docs/DATA_REQUIREMENTS.md).  Nodes without a label entry are *unlabelled*: they stay in the graph but
are never used for training, validation or testing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .tikg import TIKG, EntityType


@dataclass
class LabelSpace:
    """Global multi-label space with per-entity-type valid label sets (Table 7: #Class per type)."""
    classes: Dict[EntityType, List[str]]     # per type, ordered
    names: List[str]                          # global label columns "type:class"
    Y: np.ndarray                             # N x K  {0,1}
    valid: np.ndarray                         # N x K  bool: label k exists for node's type
    labelled: np.ndarray                      # N bool

    @property
    def K(self) -> int:
        return len(self.names)

    def type_columns(self, t: EntityType) -> np.ndarray:
        pre = f"{t.value}:"
        return np.array([i for i, n in enumerate(self.names) if n.startswith(pre)], dtype=int)

    def primary_label(self) -> np.ndarray:
        """Stratification key: rarest positive column per node (-1 if none)."""
        freq = (self.Y * self.valid).sum(0)
        out = np.full(self.Y.shape[0], -1)
        for i in np.flatnonzero(self.labelled):
            pos = np.flatnonzero(self.Y[i] > 0)
            if len(pos):
                out[i] = pos[np.argmin(freq[pos])]
        return out


def build_label_space(tikg: TIKG, assignments: Mapping[str, Sequence[str]],
                      classes: Optional[Mapping[str, Sequence[str]]] = None, min_support: int = 1) -> LabelSpace:
    """`assignments`: entity id -> list of class names. Class vocabulary per entity type is derived from the data
    unless `classes` (type value -> list) is given."""
    unknown = [k for k in assignments if k not in tikg.index]
    if unknown:
        raise ValueError(f"labels reference unknown entity ids, e.g. {unknown[:3]}")
    per_type: Dict[EntityType, Dict[str, int]] = {t: {} for t in EntityType}
    for eid, labs in assignments.items():
        t = tikg.entities[tikg.index[eid]].type
        for l in labs:
            per_type[t][l] = per_type[t].get(l, 0) + 1
    cls: Dict[EntityType, List[str]] = {}
    for t in EntityType:
        if classes and t.value in classes:
            cls[t] = list(classes[t.value])
        else:
            cls[t] = sorted(c for c, n in per_type[t].items() if n >= min_support)
    names = [f"{t.value}:{c}" for t in EntityType for c in cls[t]]
    col = {n: i for i, n in enumerate(names)}
    Y = np.zeros((tikg.n, len(names)), dtype=np.float32)
    valid = np.zeros_like(Y, dtype=bool)
    labelled = np.zeros(tikg.n, dtype=bool)
    for t in EntityType:
        for c in cls[t]:
            valid[tikg.type_nodes[t], col[f"{t.value}:{c}"]] = True
    for eid, labs in assignments.items():
        i = tikg.index[eid]
        labelled[i] = True
        t = tikg.entities[i].type
        for l in labs:
            n = f"{t.value}:{l}"
            if n in col:
                Y[i, col[n]] = 1.0
    return LabelSpace(cls, names, Y, valid, labelled)


def load_labels(tikg: TIKG, path: str | Path, **kw) -> LabelSpace:
    """JSON: {"labels": {entity_id: [class,...]}, "classes": {entity_type: [class,...]} (optional)}."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return build_label_space(tikg, d["labels"], classes=d.get("classes"), **kw)


def load_severity(tikg: TIKG, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """JSON {entity_id: score}. Returns (severity in [0,1] per node, known mask).

    CVSS base scores (0-10) are divided by 10.  Values already in [0,1] must be given as
    {"scale": 1.0, "scores": {...}}.  Unknown nodes get severity 0 and known=False (never imputed).
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    scale = 10.0
    if "scores" in d:
        scale = float(d.get("scale", 10.0))
        d = d["scores"]
    sev = np.zeros(tikg.n, dtype=np.float32)
    known = np.zeros(tikg.n, dtype=bool)
    for k, v in d.items():
        if k in tikg.index:
            sev[tikg.index[k]] = float(np.clip(float(v) / scale, 0.0, 1.0))
            known[tikg.index[k]] = True
    return sev, known


@dataclass
class ReferencePath:
    case_id: str
    nodes: List[str]
    relations: List[str]          # optional; len == len(nodes)-1 when given
    campaign: str = ""            # case scope: entities of this campaign (or `case_nodes`)
    case_nodes: Optional[List[str]] = None
    directions: Optional[List[str]] = None    # optional "fwd"/"rev" per edge (file "steps"[i]["traversal"]); EVALUATION ONLY


def load_reference_paths(path: str | Path) -> List[ReferencePath]:
    """JSON list: [{"case_id","nodes":[ids],"relations":[rel,...],"campaign":..,"case_nodes":[..]}]."""
    out = []
    for r in json.loads(Path(path).read_text(encoding="utf-8")):
        dirs = None
        if r.get("steps"):
            dirs = ["fwd" if s.get("traversal") == "forward" else "rev" for s in r["steps"]]
        out.append(ReferencePath(str(r["case_id"]), list(r["nodes"]), list(r.get("relations", [])),
                                 str(r.get("campaign", "")), r.get("case_nodes"), dirs))
    return out
