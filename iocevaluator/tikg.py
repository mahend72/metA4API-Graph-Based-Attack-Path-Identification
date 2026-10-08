"""Threat Intelligence Knowledge Graph (TIKG) - manuscript Sec. 3.2.

Implements the 7 entity types (Table 1), the 9 relation names, the 11 canonical
triplet signatures T1-T11 (Table 2) and a loader/validator.  Inverse directions
are *not* stored; they are produced on demand via transposition (manuscript
"Triplets" paragraph).

One documented extension: ``X1 = <TA, uses, F>``.  The manuscript's meta-graphs
chi_10/11/13/18 relate actors to domains/IPs/e-mails/hashes, but Table 2 has no
actor->file triplet.  We add it (relation ``uses``) and report it as a
manuscript inconsistency (see docs/MANUSCRIPT_MAPPING.md).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)


class EntityType(str, Enum):
    THREAT_ACTOR = "threat_actor"
    VULNERABILITY = "vulnerability"
    ATTACK_METHOD = "attack_method"
    FILE = "file"  # File / Artefact (hashes, domains, IPs, e-mails, URLs ...)
    ATTACK_TYPE = "attack_type"
    DEVICE = "device"
    PLATFORM = "platform"


ABBR: Dict[EntityType, str] = {
    EntityType.THREAT_ACTOR: "TA",
    EntityType.VULNERABILITY: "V",
    EntityType.ATTACK_METHOD: "M",
    EntityType.FILE: "F",
    EntityType.ATTACK_TYPE: "AT",
    EntityType.DEVICE: "D",
    EntityType.PLATFORM: "P",
}
FROM_ABBR = {v: k for k, v in ABBR.items()}

# The nine relation types R (Sec. 3.2.2).  `enables` is in R but no triplet in Table 2 uses it.
RELATIONS: Tuple[str, ...] = (
    "unauthorised_access", "assists", "runs_on", "exploits", "evolves_to",
    "enables", "contains", "uses", "affected_by",
)

TA, V, M, F, AT, D, P = (EntityType.THREAT_ACTOR, EntityType.VULNERABILITY, EntityType.ATTACK_METHOD,
                         EntityType.FILE, EntityType.ATTACK_TYPE, EntityType.DEVICE, EntityType.PLATFORM)

# Table 2 (T1-T11) + documented extension X1.
TRIPLET_SIGNATURES: Dict[str, Tuple[EntityType, str, EntityType]] = {
    "T1": (TA, "exploits", V),
    "T2": (D, "affected_by", V),
    "T3": (P, "affected_by", V),
    "T4": (F, "contains", V),
    "T5": (AT, "exploits", V),
    "T6": (V, "evolves_to", V),
    "T7": (D, "runs_on", P),
    "T8": (TA, "unauthorised_access", D),
    "T9": (TA, "assists", TA),
    "T10": (M, "exploits", V),
    "T11": (TA, "uses", M),
    "X1": (TA, "uses", F),  # extension, see module docstring
}
_SIG_BY_TRIPLE = {v: k for k, v in TRIPLET_SIGNATURES.items()}

FILE_SUBTYPES = ("hash", "domain", "ip", "email", "url", "hostname", "other")


@dataclass
class Entity:
    id: str
    type: EntityType
    name: str = ""
    subtype: str = ""            # for File: one of FILE_SUBTYPES
    campaign: str = ""           # group id used for campaign-isolated CV
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Triplet:
    s: int
    r: str
    t: int
    sig: str


class TIKGError(ValueError):
    pass


class TIKG:
    """Typed, directed knowledge graph KG = {E, R, T}."""

    def __init__(self, entities: Sequence[Entity], triplets: Iterable[Tuple[str, str, str]], strict: bool = True):
        self.entities: List[Entity] = list(entities)
        self.index: Dict[str, int] = {}
        for i, e in enumerate(self.entities):
            if e.id in self.index:
                raise TIKGError(f"duplicate entity id: {e.id!r}")
            self.index[e.id] = i
        self.n = len(self.entities)
        self.types = np.array([e.type.value for e in self.entities])
        self.type_nodes: Dict[EntityType, np.ndarray] = {
            t: np.flatnonzero(self.types == t.value) for t in EntityType
        }
        self._local = np.full(self.n, -1, dtype=np.int64)
        for t, idx in self.type_nodes.items():
            self._local[idx] = np.arange(len(idx))
        self.triplets: List[Triplet] = []
        seen: Set[Tuple[int, str, int]] = set()
        for s, r, t in triplets:
            if s not in self.index or t not in self.index:
                raise TIKGError(f"triplet references unknown entity: {(s, r, t)}")
            si, ti = self.index[s], self.index[t]
            key = (self.entities[si].type, r, self.entities[ti].type)
            sig = _SIG_BY_TRIPLE.get(key)
            if sig is None:
                if strict:
                    raise TIKGError(f"triplet {(s, r, t)} has signature {tuple(getattr(k, 'value', k) for k in key)} "
                                    "not in Table 2 (T1-T11) or extension X1")
                continue
            if (si, r, ti) in seen:
                continue
            seen.add((si, r, ti))
            self.triplets.append(Triplet(si, r, ti, sig))

    # ------------------------------------------------------------------ basic info
    def local(self, i: int) -> int:
        return int(self._local[i])

    def subtype_mask(self, subtypes: Optional[Sequence[str]]) -> np.ndarray:
        """Boolean mask over *all* nodes: File nodes whose subtype is in `subtypes`."""
        if subtypes is None:
            return np.ones(self.n, dtype=bool)
        st = {s.lower() for s in subtypes}
        return np.array([e.subtype.lower() in st for e in self.entities])

    def campaigns(self) -> np.ndarray:
        return np.array([e.campaign for e in self.entities], dtype=object)

    def stats(self) -> Dict[str, Any]:
        deg = np.zeros(self.n)
        for t in self.triplets:
            deg[t.s] += 1
            deg[t.t] += 1
        return {
            "n_nodes": self.n,
            "n_relations": len(self.triplets),
            "avg_undirected_degree": float(2 * len(self.triplets) / max(self.n, 1)),
            "nodes_per_type": {t.value: int(len(v)) for t, v in self.type_nodes.items()},
            "triplets_per_signature": {k: sum(1 for t in self.triplets if t.sig == k) for k in TRIPLET_SIGNATURES},
            "n_campaigns": int(len({e.campaign for e in self.entities if e.campaign})),
            "isolated_nodes": int((deg == 0).sum()),
        }

    # ------------------------------------------------------------------ matrices
    def Q(self, src: EntityType, rel: str, tgt: EntityType, *, src_subtypes: Optional[Sequence[str]] = None,
          tgt_subtypes: Optional[Sequence[str]] = None, exclude: Optional[np.ndarray] = None) -> sp.csr_matrix:
        """Binary biadjacency matrix Q_{(src,tgt)} of shape (|src nodes|, |tgt nodes|).

        `exclude` is a boolean mask over all nodes; triplets touching an excluded node are dropped
        (used to build the leakage-safe per-fold training graph).
        """
        rows, cols = [], []
        sm = self.subtype_mask(src_subtypes)
        tm = self.subtype_mask(tgt_subtypes)
        for t in self.triplets:
            if self.entities[t.s].type != src or t.r != rel or self.entities[t.t].type != tgt:
                continue
            if exclude is not None and (exclude[t.s] or exclude[t.t]):
                continue
            if not (sm[t.s] and tm[t.t]):
                continue
            rows.append(self._local[t.s])
            cols.append(self._local[t.t])
        shape = (len(self.type_nodes[src]), len(self.type_nodes[tgt]))
        m = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape)
        return m

    def observed_adjacency(self, symmetric: bool = True, exclude: Optional[np.ndarray] = None) -> sp.csr_matrix:
        """Binary N x N adjacency of the *observed* triplets (the non-semantic baseline)."""
        r, c = [], []
        for t in self.triplets:
            if exclude is not None and (exclude[t.s] or exclude[t.t]):
                continue
            r.append(t.s)
            c.append(t.t)
        a = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(self.n, self.n))
        if symmetric:
            a = a.maximum(a.T)
        a.setdiag(0)
        a.eliminate_zeros()
        return a.tocsr()

    def relation_adjacencies(self) -> Dict[str, sp.csr_matrix]:
        """Per-triplet-signature symmetric adjacency (used by the HGT-style baseline)."""
        out: Dict[str, Tuple[List[int], List[int]]] = {}
        for t in self.triplets:
            rr, cc = out.setdefault(t.sig, ([], []))
            rr.append(t.s)
            cc.append(t.t)
        return {k: sp.csr_matrix((np.ones(len(v[0]), dtype=np.float32), v), shape=(self.n, self.n))
                for k, v in out.items()}

    # ------------------------------------------------------------------ IO
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [
                {"id": e.id, "type": e.type.value, "name": e.name, "subtype": e.subtype,
                 "campaign": e.campaign, "attrs": e.attrs} for e in self.entities
            ],
            "triplets": [{"s": self.entities[t.s].id, "r": t.r, "t": self.entities[t.t].id} for t in self.triplets],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")


def load_tikg(path: str | Path, strict: bool = True) -> TIKG:
    """Load the native TIKG JSON format (see docs/DATA_REQUIREMENTS.md)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return tikg_from_dict(d, strict=strict)


def tikg_from_dict(d: Dict[str, Any], strict: bool = True) -> TIKG:
    ents = []
    for e in d["entities"]:
        try:
            et = EntityType(e["type"])
        except ValueError as ex:
            raise TIKGError(f"unknown entity type {e['type']!r} (valid: {[t.value for t in EntityType]})") from ex
        ents.append(Entity(id=str(e["id"]), type=et, name=e.get("name", ""), subtype=e.get("subtype", ""),
                           campaign=str(e.get("campaign", "")), attrs=dict(e.get("attrs", {}))))
    trips = []
    for t in d["triplets"]:
        trips.append((str(t["s"]), t["r"], str(t["t"])) if isinstance(t, dict) else (str(t[0]), t[1], str(t[2])))
    return TIKG(ents, trips, strict=strict)


# ---------------------------------------------------------------------- OTX / ThreatRecord adapter
def _file_subtype(ind_type: str) -> Optional[str]:
    t = (ind_type or "").strip().lower()
    if t.startswith("filehash"):
        return "hash"
    return {"ipv4": "ip", "ipv6": "ip", "ip": "ip", "domain": "domain", "hostname": "hostname",
            "url": "url", "email": "email"}.get(t)


def tikg_from_threats(threats: Iterable[Any], extra_triplets: Optional[Iterable[Tuple[str, str, str]]] = None,
                      extra_entities: Optional[Iterable[Entity]] = None) -> TIKG:
    """Build a TIKG from normalised OTX ``ThreatRecord`` objects.

    Only relations that are *explicit* in the feed are created:
      * actor --uses--> File(hash/domain/ip/email/url/hostname)   [X1]
      * actor --exploits--> Vulnerability (CVE indicators)          [T1]
      * actor --uses--> Attack method (ATT&CK ids)                  [T11]
    Devices, platforms, attack types and every other triplet (T2-T10) are **not** inferred - OTX does not
    carry them.  Supply them via ``extra_entities`` / ``extra_triplets`` (e.g. an asset inventory or an NVD/CPE
    join).  ``campaign`` is the id of the first pulse in which an entity appears.
    """
    ents: Dict[str, Entity] = {}
    trips: List[Tuple[str, str, str]] = []

    def add(eid: str, et: EntityType, name: str, campaign: str, subtype: str = "", created=None, pulse=None) -> str:
        e = ents.get(eid)
        if e is None:
            e = Entity(id=eid, type=et, name=name, subtype=subtype, campaign=campaign,
                       attrs={"update_count": 0, "target_countries": [], "industries": []})
            ents[eid] = e
        e.attrs["update_count"] = int(e.attrs.get("update_count", 0)) + 1
        if created is not None:
            iso = created.isoformat() if hasattr(created, "isoformat") else str(created)
            e.attrs["first_seen"] = min(e.attrs.get("first_seen", iso), iso)
            e.attrs["last_seen"] = max(e.attrs.get("last_seen", iso), iso)
        if pulse is not None:
            for c in (getattr(pulse, "targeted_countries", None) or []):
                if c and c not in e.attrs["target_countries"]:
                    e.attrs["target_countries"].append(c)
            for c in (getattr(pulse, "industries", None) or []):
                if c and c not in e.attrs["industries"]:
                    e.attrs["industries"].append(c)
        return eid

    for t in threats:
        actor = (t.adversary or "").strip()
        if not actor or actor.lower() in {"unknown", "n/a"}:
            continue
        camp = t.name
        a = add(f"actor:{actor}", TA, actor, camp, created=t.created, pulse=t)
        for ind in t.indicators:
            val = (ind.indicator or "").strip()
            if not val:
                continue
            if (ind.type or "").strip().lower() == "cve":
                v = add(f"vuln:{val.upper()}", V, val.upper(), camp, created=t.created, pulse=t)
                trips.append((a, "exploits", v))
                continue
            st = _file_subtype(ind.type)
            if st:
                f = add(f"file:{st}:{val.lower()}", F, val, camp, subtype=st, created=t.created, pulse=t)
                trips.append((a, "uses", f))
        for aid in t.attack_ids or []:
            if aid:
                m = add(f"method:{aid}", M, aid, camp, created=t.created, pulse=t)
                trips.append((a, "uses", m))
    for e in extra_entities or []:
        ents.setdefault(e.id, e)
    trips.extend(extra_triplets or [])
    return TIKG(list(ents.values()), trips, strict=True)
