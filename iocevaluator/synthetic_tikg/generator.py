"""Campaign-conditioned generator for a SYNTHETIC Threat Intelligence Knowledge Graph.

PURPOSE: development and end-to-end pipeline testing only.  Nothing produced here is, or may be reported as, a
reproduction of the manuscript's real AlienVault-OTX experiments.

Generative story (all randomness comes from per-stage streams derived from ``seed``):
  1. Latent *families* (threat behaviours, ``vocab.FAMILIES``) and *campaigns* (several per family).  Campaigns of the
     same family are *related*: they share device/platform mix, target sectors/countries, weakness vocabulary.
  2. Entities are created per campaign (campaign sizes are uneven) -> every node has one primary ``campaign``
     (the group id for campaign-isolated CV) and one ``family``.
  3. Edges are sampled to exact per-triplet quotas (T1-T11 only, Table 2) with locality: mostly inside the source's
     campaign, sometimes inside a *related* campaign (this is what creates shared vulnerabilities / methods /
     platforms across related campaigns), rarely across families.  Popularity weights give heavy-tailed degrees.
  4. Multi-label targets are drawn from campaign-specific class distributions anchored on the family.
  5. Reference attack paths are random instantiations of schema-valid, kill-chain-ordered templates restricted to
     one campaign, so every hop is an existing stored edge.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..tikg import RELATIONS, TRIPLET_SIGNATURES
from . import vocab as V
from .cve_assign import assign_cves
from .cve_pool import load_pool, select_cves
from .profiles import PROFILES, TYPES, Profile

GENERATOR_VERSION = "1.0"
MANUSCRIPT_VULN = 2298
STAGES = {"campaigns": 1, "nodes": 2, "edges": 3, "labels": 4, "paths": 5}
BASE_DATE = datetime(2022, 1, 1)

ABBR = {"TA": "threat_actor", "V": "vulnerability", "M": "attack_method", "F": "file", "AT": "attack_type",
        "D": "device", "P": "platform"}
ID_PREFIX = {"threat_actor": "actor", "vulnerability": "vuln", "attack_method": "method", "file": "file",
             "attack_type": "attack_type", "device": "device", "platform": "platform"}
CLASS_PREFIX = {"threat_actor": "ta", "vulnerability": "vu", "attack_method": "am", "file": "fi",
                "attack_type": "at", "device": "dv", "platform": "pl"}
FILE_SUBTYPES = ("hash", "domain", "ip", "email", "url", "hostname")
FILE_SUBTYPE_P = (0.58, 0.15, 0.09, 0.06, 0.06, 0.06)

# (relation, source type, target type) per canonical triplet; X1 is the optional extension (default OFF).
SIGS: Dict[str, Tuple[str, str, str]] = {k: (s.value, r, t.value) for k, (s, r, t) in TRIPLET_SIGNATURES.items()}
CORE_SIGS = tuple(f"T{i}" for i in range(1, 12))

# P(target in same campaign, in a related campaign of the same family, in another family)
DEFAULT_LOCALITY = (0.70, 0.26, 0.04)
LOCALITY = {"T6": (0.78, 0.20, 0.02), "T9": (0.75, 0.22, 0.03), "T7": (0.50, 0.40, 0.10),
            "T3": (0.60, 0.32, 0.08), "T11": (0.65, 0.30, 0.05)}
COVER_SIG = {"threat_actor": "T1", "attack_method": "T10", "attack_type": "T5", "file": "T4", "platform": "T3",
             "device": "T2"}
V_INCOMING = ("T1", "T2", "T3", "T4", "T5", "T10")


# ------------------------------------------------------------------------------------------------ attack-path templates
_TOKEN = re.compile(r"^(?:-(\w+)>|<(\w+)-)$")


def parse_template(spec: str) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]:
    """'TA -uses> M -exploits> V <affected_by- D' -> (types, ((rel, 'f'|'r'), ...)).  'f': the stored edge goes
    current -> next; 'r': it goes next -> current."""
    toks = spec.split()
    nodes, steps = toks[0::2], []
    for c in toks[1::2]:
        m = _TOKEN.match(c)
        steps.append((m.group(1), "f") if m.group(1) else (m.group(2), "r"))
    return tuple(ABBR[n] for n in nodes), tuple(steps)


# Kill-chain-ordered: actor/method/attack-type -> vulnerability -> affected asset -> platform.
PATH_TEMPLATES: Dict[int, List[str]] = {
    2: ["TA -uses> M -exploits> V", "TA -exploits> V <affected_by- D", "TA -exploits> V <affected_by- P",
        "TA -exploits> V <contains- F", "TA -unauthorised_access> D -affected_by> V",
        "M -exploits> V <affected_by- D", "AT -exploits> V <affected_by- D", "TA -assists> TA -exploits> V",
        "TA -exploits> V -evolves_to> V"],
    3: ["TA -uses> M -exploits> V <affected_by- D", "TA -exploits> V <affected_by- D -runs_on> P",
        "TA -assists> TA -exploits> V <affected_by- D", "TA -uses> M -exploits> V -evolves_to> V",
        "TA -unauthorised_access> D -affected_by> V <contains- F", "TA -unauthorised_access> D -runs_on> P -affected_by> V",
        "TA -exploits> V -evolves_to> V <affected_by- D", "AT -exploits> V <affected_by- D -runs_on> P"],
    4: ["TA -uses> M -exploits> V <affected_by- D -runs_on> P", "TA -assists> TA -uses> M -exploits> V <affected_by- D",
        "TA -uses> M -exploits> V -evolves_to> V <affected_by- D",
        "TA -unauthorised_access> D -affected_by> V -evolves_to> V <contains- F",
        "TA -assists> TA -uses> M -exploits> V <affected_by- D -runs_on> P",
        "TA -uses> M -exploits> V -evolves_to> V <affected_by- D -runs_on> P",
        "TA -assists> TA -uses> M -exploits> V -evolves_to> V <affected_by- D -runs_on> P"],
}
_SIG_LOOKUP = {(s, r, t): k for k, (s, r, t) in SIGS.items()}


def _check_templates() -> None:
    for specs in PATH_TEMPLATES.values():
        for sp in specs:
            types, steps = parse_template(sp)
            for i, (rel, d) in enumerate(steps):
                a, b = (types[i], types[i + 1]) if d == "f" else (types[i + 1], types[i])
                if (a, rel, b) not in _SIG_LOOKUP or _SIG_LOOKUP[(a, rel, b)] not in CORE_SIGS:
                    raise AssertionError(f"template {sp!r} step {i} is not a Table-2 triplet")


_check_templates()


# ------------------------------------------------------------------------------------------------ small helpers
def _rng(seed: int, stage: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), STAGES[stage]]))


def _allocate(total: int, weights: Sequence[float]) -> List[int]:
    """Split `total` over len(weights) bins (>=1 each) proportionally to `weights` (largest remainder)."""
    n = len(weights)
    if total < n:
        raise ValueError(f"cannot give {n} campaigns at least one node each from {total} nodes")
    w = np.asarray(weights, float)
    w = w / w.sum()
    raw = w * (total - n)
    base = np.floor(raw).astype(int)
    rem = (total - n) - int(base.sum())
    base[np.argsort(-(raw - base), kind="stable")[:rem]] += 1
    return (base + 1).tolist()


def _iso(day: float) -> str:
    return (BASE_DATE + timedelta(days=float(day))).isoformat(timespec="seconds")


class _Pool:
    """Popularity-weighted sampler over a fixed set of node indices."""

    def __init__(self, idx: Sequence[int], w: np.ndarray):
        self.idx = np.asarray(idx, dtype=np.int64)
        c = np.cumsum(np.asarray(w, float))
        self.cum = c / c[-1]

    def draw(self, rng: np.random.Generator) -> int:
        j = int(np.searchsorted(self.cum, rng.random(), side="right"))
        return int(self.idx[min(j, len(self.idx) - 1)])


# ------------------------------------------------------------------------------------------------ CVSS v3.1
_AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
_AC = {"L": 0.77, "H": 0.44}
_PR_U = {"N": 0.85, "L": 0.62, "H": 0.27}
_PR_C = {"N": 0.85, "L": 0.68, "H": 0.5}
_UI = {"N": 0.85, "R": 0.62}
_CIA = {"H": 0.56, "L": 0.22, "N": 0.0}


def _roundup(x: float) -> float:
    i = round(x * 100000)
    return i / 100000.0 if i % 10000 == 0 else (math.floor(i / 10000) + 1) / 10.0


def cvss31_base_score(vector: str) -> float:
    m = dict(p.split(":") for p in vector.split("/")[1:])
    changed = m["S"] == "C"
    iss = 1 - (1 - _CIA[m["C"]]) * (1 - _CIA[m["I"]]) * (1 - _CIA[m["A"]])
    impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15 if changed else 6.42 * iss
    expl = 8.22 * _AV[m["AV"]] * _AC[m["AC"]] * (_PR_C if changed else _PR_U)[m["PR"]] * _UI[m["UI"]]
    if impact <= 0:
        return 0.0
    return _roundup(min((1.08 if changed else 1.0) * (impact + expl), 10.0))


def _cvss_vector(rng: np.random.Generator, fam: dict) -> str:
    av = "N" if rng.random() < fam["p_net"] else str(rng.choice(["A", "L", "P"], p=[0.3, 0.5, 0.2]))
    ac = str(rng.choice(["L", "H"], p=[0.8, 0.2]))
    pr = str(rng.choice(["N", "L", "H"], p=[0.5, 0.35, 0.15]))
    ui = str(rng.choice(["N", "R"], p=[0.7, 0.3]))
    s = str(rng.choice(["U", "C"], p=[0.8, 0.2]))
    hi = fam["impact_hi"]
    cia = [str(rng.choice(["H", "L", "N"], p=[hi, (1 - hi) * 0.6, (1 - hi) * 0.4])) for _ in range(3)]
    if cia == ["N", "N", "N"]:
        cia[0] = "L"
    return f"CVSS:3.1/AV:{av}/AC:{ac}/PR:{pr}/UI:{ui}/S:{s}/C:{cia[0]}/I:{cia[1]}/A:{cia[2]}"


# ------------------------------------------------------------------------------------------------ dataset container
@dataclass
class SyntheticDataset:
    nodes: List[dict]
    edges: List[dict]
    labels: dict
    campaigns: dict
    attack_paths: List[dict]
    metadata: dict

    FILES = ("nodes", "edges", "labels", "campaigns", "attack_paths", "metadata")

    def payloads(self) -> Dict[str, Any]:
        return {f"{k}.json": getattr(self, k) for k in self.FILES}


def dumps(obj: Any) -> str:
    def conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    return json.dumps(obj, indent=1, ensure_ascii=False, default=conv) + "\n"


# ------------------------------------------------------------------------------------------------ generator
class _Builder:
    def __init__(self, profile: Profile, seed: int, include_x1: bool, cves: Optional[List[dict]] = None,
                 snapshot: Optional[dict] = None):
        self.p, self.seed, self.include_x1 = profile, int(seed), include_x1
        self.cves, self.snapshot = cves, snapshot                 # real-CVE (semi-synthetic) mode iff cves is given
        self.real = cves is not None
        self.nodes: List[dict] = []
        self.fam: List[int] = []
        self.camp: List[int] = []
        self.kind: List[str] = []
        self.born: List[float] = []
        self.pop: List[float] = []
        self.type_idx: Dict[str, List[int]] = {t: [] for t in TYPES}

    # ---------------------------------------------------------------- 1. campaigns
    def build_campaigns(self) -> None:
        rng = _rng(self.seed, "campaigns")
        F, C = self.p.n_families, self.p.n_campaigns
        if F > len(V.FAMILIES):
            raise ValueError("not enough family templates")
        self.fam_defs = V.FAMILIES[:F]
        self.camps: List[dict] = []
        for c in range(C):
            f = c % F
            fd = self.fam_defs[f]
            pick = lambda xs: [str(x) for x in rng.choice(xs, size=int(rng.integers(1, len(xs) + 1)), replace=False)]
            start = float(rng.integers(0, 1100))
            self.camps.append(dict(
                id=f"camp-{c + 1:03d}", index=c, family=f, start=start, duration=float(rng.integers(60, 300)),
                tgt=pick(fd["tgt"]), src=pick(fd["src"]), ind=pick(fd["ind"]),
                scale=float(np.exp(rng.normal(0, 0.45))),
                codename=f"{V.ACTOR_ADJ[int(rng.integers(len(V.ACTOR_ADJ)))]} {V.ACTOR_NOUN[int(rng.integers(len(V.ACTOR_NOUN)))]}"))
        self.alloc: Dict[str, List[int]] = {}
        for t in TYPES:
            w = [c["scale"] * float(np.exp(rng.normal(0, 0.25))) for c in self.camps]
            self.alloc[t] = _allocate(self.p.node_counts[t], w)
        if self.real:
            self.camp_cves = assign_cves(self.cves, self.alloc["vulnerability"], [c["family"] for c in self.camps],
                                         self.fam_defs, self.seed)

    # ---------------------------------------------------------------- 2. nodes
    def _add_node(self, t: str, nid: str, name: str, subtype: str, c: dict, rng, text: str, kind: str = "",
                  extra: Optional[dict] = None) -> int:
        first = c["start"] + float(rng.uniform(0, 0.6 * c["duration"]))
        last = min(first + c["duration"] * float(rng.uniform(0.1, 0.5)), c["start"] + c["duration"] + 30)
        lam = {"threat_actor": 4, "vulnerability": 2}.get(t, 2)
        attrs = {"synthetic": True, "family": self.fam_defs[c["family"]]["key"], "text": text,
                 "first_seen": _iso(first), "last_seen": _iso(last), "update_count": int(1 + rng.poisson(lam)),
                 "source_countries": list(c["src"]), "target_countries": list(c["tgt"]), "industries": list(c["ind"])}
        attrs.update(extra or {})
        i = len(self.nodes)
        node = {"id": nid, "type": t, "name": name, "subtype": subtype, "campaign": c["id"], "attrs": attrs}
        if self.real:
            node["provenance"] = "synthetic"
        self.nodes.append(node)
        self.fam.append(c["family"])
        self.camp.append(c["index"])
        self.kind.append(kind)
        self.born.append(first)
        sigma = 1.0 if t == "vulnerability" else 0.8
        self.pop.append(float(np.exp(rng.normal(0, sigma))))
        self.type_idx[t].append(i)
        return i

    def _add_real_cve(self, r: dict, c: dict, rng) -> int:
        """Real public CVE metadata; campaign membership is the only synthetic part.  No timing/country/industry
        facts are attached to the CVE (that would be an unsupported claim about a real vulnerability)."""
        attrs = {"synthetic": False, "provenance": "real_public", "text": r["description"],
                 "first_seen": r["published"], "last_seen": r["last_modified"], "cve_id": r["cve_id"],
                 "published": r["published"], "last_modified": r["last_modified"], "vuln_status": r["vuln_status"],
                 "source_identifier": r["source_identifier"], "cvss_version": r["cvss_version"],
                 "cvss_base_score": r["cvss_base_score"], "cvss_vector": r["cvss_vector"], "severity": r["severity"],
                 "cwe": r["cwe"], "vendors": r["vendors"], "products": r["products"], "cpe": r["cpe"],
                 "platform_parts": r["platform_parts"], "references": r["references"],
                 "synthetic_fields": [], "cve_snapshot_sha256": self.snapshot["sha256"]}
        i = len(self.nodes)
        self.nodes.append({"id": f"vuln:{r['cve_id']}", "type": "vulnerability", "name": r["cve_id"], "subtype": "",
                           "campaign": c["id"], "provenance": "real_public", "attrs": attrs})
        self.fam.append(c["family"])
        self.camp.append(c["index"])
        self.kind.append("")
        self.born.append((datetime.fromisoformat(r["published"]) - BASE_DATE).total_seconds() / 86400.0)
        self.pop.append(float(np.exp(rng.normal(0, 1.0))))
        self.type_idx["vulnerability"].append(i)
        return i

    def build_nodes(self) -> None:
        rng = _rng(self.seed, "nodes")
        s = self.seed
        used_names: set = set()

        def unique(name: str) -> str:
            n, k = name, 2
            while n in used_names:
                n, k = f"{name} {k}", k + 1
            used_names.add(n)
            return n

        # name pools
        combos = [f"{a} {n}{x}" for a in V.ACTOR_ADJ for n in V.ACTOR_NOUN for x in V.ACTOR_SUFFIX]
        actor_names = [combos[j] for j in rng.permutation(len(combos))]
        ips = [f"{net}.{k}" for net in ("192.0.2", "198.51.100", "203.0.113") for k in range(1, 255)]
        ips = [ips[j] for j in rng.permutation(len(ips))]
        cve_seen: set = set()
        tech_ids = [f"T{1000 + k}" for k in rng.permutation(700)]
        counters = {t: 0 for t in TYPES}
        k_in_c = [0] * len(self.camps)

        for c in self.camps:
            fd = self.fam_defs[c["family"]]
            theme = fd["title"]
            for t in TYPES:
                for _ in range(self.alloc[t][c["index"]]):
                    k = counters[t] = counters[t] + 1
                    if t == "threat_actor":
                        name = actor_names[k - 1] if k <= len(actor_names) else f"{actor_names[k % len(actor_names)]} {k}"
                        trade = " and ".join(str(x) for x in rng.choice(fd["tradecraft"], size=2, replace=False))
                        text = (f"{name} is a fictional threat group profile active against "
                                f"{', '.join(c['ind'])} organisations in {', '.join(c['tgt'])}. "
                                f"Reported tradecraft: {trade}. Behaviour theme: {theme}.")
                        self._add_node(t, f"actor:SYN-TA-{k:04d}", name, "", c, rng, text,
                                       extra={"alias": f"SYN-TA-{k:04d}"})
                    elif t == "vulnerability" and self.real:
                        self._add_real_cve(self.camp_cves[c["index"]][k_in_c[c["index"]]], c, rng)
                        k_in_c[c["index"]] += 1
                    elif t == "vulnerability":
                        while True:
                            cid = f"CVE-{int(rng.integers(2090, 2100))}-{int(rng.integers(1, 100000)):05d}"
                            if cid not in cve_seen:
                                cve_seen.add(cid)
                                break
                        dk = str(rng.choice(list(fd["devices"]), p=_norm(fd["devices"])))
                        pk = str(rng.choice(list(fd["platforms"]), p=_norm(fd["platforms"])))
                        product = V.DEVICE_KIND_LABEL[dk] if rng.random() < 0.5 else V.PLATFORM_KIND_LABEL[pk]
                        weak = str(rng.choice(fd["weakness"]))
                        vec = _cvss_vector(rng, fd)
                        ver = f"{int(rng.integers(1, 6))}.{int(rng.integers(0, 10))}.{int(rng.integers(0, 20))}"
                        text = f"{cid}: {weak} in {product} version {ver} allows an attacker to affect production systems."
                        self._add_node(t, f"vuln:{cid}", cid, "", c, rng, text,
                                       extra={"cvss_version": "3.1", "cvss_vector": vec,
                                              "cvss_base_score": cvss31_base_score(vec), "affected_version": ver})
                    elif t == "attack_method":
                        tid = tech_ids[k - 1]
                        if rng.random() < 0.4:
                            tid += f".{int(rng.integers(1, 6)):03d}"
                        verb = V.TECHNIQUE_VERBS[int(rng.integers(len(V.TECHNIQUE_VERBS)))]
                        name = f"{verb} (synthetic {tid})"
                        text = (f"Synthetic technique {tid}: {verb.lower()}, observed in {theme} activity "
                                f"against {', '.join(c['ind'])} targets.")
                        self._add_node(t, f"method:{tid}", name, "", c, rng, text, extra={"technique_id": tid})
                    elif t == "attack_type":
                        base = V.ATTACK_TYPE_BASE[int(rng.integers(len(V.ATTACK_TYPE_BASE)))]
                        qual = V.ATTACK_TYPE_QUAL[int(rng.integers(len(V.ATTACK_TYPE_QUAL)))]
                        name = unique(f"{base} affecting {qual}")
                        text = f"Incident category: {name.lower()}, associated with {theme}."
                        self._add_node(t, f"attack_type:SYN-AT-{k:04d}", name, "", c, rng, text)
                    elif t == "device":
                        dk = str(rng.choice(list(fd["devices"]), p=_norm(fd["devices"])))
                        vendor = V.VENDORS[int(rng.integers(len(V.VENDORS)))]
                        model = f"{vendor.split()[0][:2].upper()}-{int(rng.integers(100, 9999))}"
                        name = unique(f"{vendor} {model} {V.DEVICE_KIND_LABEL[dk]}")
                        text = (f"Fictional {V.DEVICE_KIND_LABEL[dk]} from {vendor}, deployed in "
                                f"{c['ind'][0]} production cells; exposed to {theme}.")
                        self._add_node(t, f"device:SYN-DEV-{k:04d}", name, "", c, rng, text, kind=dk,
                                       extra={"vendor": vendor, "model": model, "device_kind": dk})
                    elif t == "platform":
                        pk = str(rng.choice(list(fd["platforms"]), p=_norm(fd["platforms"])))
                        ver = f"{int(rng.integers(1, 12))}.{int(rng.integers(0, 10))}.{int(rng.integers(0, 40))}"
                        name = unique(f"{V.PLATFORM_KIND_LABEL[pk]} {ver}")
                        text = f"Platform {name} used on additive-manufacturing assets; relevant to {theme}."
                        self._add_node(t, f"platform:SYN-PLT-{k:04d}", name, "", c, rng, text, kind=pk,
                                       extra={"platform_kind": pk, "version": ver})
                    else:  # file / artefact
                        st = str(rng.choice(FILE_SUBTYPES, p=FILE_SUBTYPE_P))
                        word = V.ACTOR_NOUN[int(rng.integers(len(V.ACTOR_NOUN)))].lower()
                        if st == "hash":
                            alg = str(rng.choice(["sha256", "md5", "sha1"], p=[0.7, 0.2, 0.1]))
                            val = hashlib.new(alg, f"syn|{s}|file|{k}".encode()).hexdigest()
                            ext = str(rng.choice(["stl", "3mf", "gcode", "exe", "dll", "bin", "zip", "ps1"]))
                            extra = {"hash_algorithm": alg, "file_name": f"{word}_{int(rng.integers(1, 99))}.{ext}",
                                     "file_size": int(rng.integers(2_000, 90_000_000))}
                        elif st == "domain":
                            val, extra = f"{word}-{k}.example", {}
                        elif st == "ip":
                            val, extra = ips[k % len(ips)] if k <= len(ips) else f"192.0.2.{k % 254 + 1}", {}
                        elif st == "email":
                            val, extra = f"{word}{k}@example.invalid", {}
                        elif st == "url":
                            val, extra = f"https://{word}-{k}.example/{int(rng.integers(1000, 9999))}/pkg", {}
                        else:
                            val, extra = f"{word}{k}.corp.invalid", {}
                        text = f"Fictional {st} indicator associated with {theme}; seen in artefacts targeting {c['ind'][0]}."
                        self._add_node(t, f"file:{st}:{val}", val, st, c, rng, text, extra=extra)
        n = len(self.nodes)
        self.fam_a, self.camp_a = np.array(self.fam), np.array(self.camp)
        self.pop_a = np.array(self.pop)
        ids = [x["id"] for x in self.nodes]
        if len(set(ids)) != n:
            raise AssertionError("duplicate node ids generated")
        self.index = {x: i for i, x in enumerate(ids)}
        self._build_pools()

    def _build_pools(self) -> None:
        self.pools: Dict[tuple, _Pool] = {}
        for t in TYPES:
            idx = np.array(self.type_idx[t])
            self.pools[(t, "a", 0)] = _Pool(idx, self.pop_a[idx])
            for f in range(self.p.n_families):
                sel = idx[self.fam_a[idx] == f]
                if len(sel):
                    self.pools[(t, "f", f)] = _Pool(sel, self.pop_a[sel])
            for c in range(self.p.n_campaigns):
                sel = idx[self.camp_a[idx] == c]
                if len(sel):
                    self.pools[(t, "c", c)] = _Pool(sel, self.pop_a[sel])

    # ---------------------------------------------------------------- 3. edges
    def _draw(self, src: int, tgt_type: str, probs: Tuple[float, float, float], pred=None) -> Optional[int]:
        rng = self.rng
        r = rng.random()
        loc = 0 if r < probs[0] else (1 if r < probs[0] + probs[1] else 2)
        for attempt in range(40):
            if attempt in (14, 28):                        # widen the search if the local pool keeps failing
                loc = min(loc + 1, 2)
            pool = self.pools.get([(tgt_type, "c", int(self.camp_a[src])), (tgt_type, "f", int(self.fam_a[src])),
                                   (tgt_type, "a", 0)][loc])
            if pool is None:
                loc = min(loc + 1, 2)
                continue
            cand = pool.draw(rng)
            if cand == src:
                continue
            if loc == 1 and self.camp_a[cand] == self.camp_a[src]:
                continue
            if loc == 2 and self.fam_a[cand] == self.fam_a[src]:
                continue
            if pred is not None and not pred(src, cand):
                continue
            return cand
        return None

    def _add_edge(self, s: int, rel: str, t: int, sig: str) -> bool:
        key = (s, rel, t)
        if key in self.seen or s == t:
            return False
        self.seen.add(key)
        if self.camp_a[s] == self.camp_a[t]:
            loc = "campaign"
        elif self.fam_a[s] == self.fam_a[t]:
            loc = "family"
        else:
            loc = "cross_family"
        edge = {"source": self.nodes[s]["id"], "relation": rel, "target": self.nodes[t]["id"],
                "triplet": sig, "locality": loc}
        if self.real:
            edge["provenance"] = "synthetic"                  # no relation involving a real CVE is a real-world claim
        self.edges.append(edge)
        self.deg[s] += 1
        self.deg[t] += 1
        self.count[sig] += 1
        self.out_adj.setdefault((s, rel, "f"), []).append(t)
        self.out_adj.setdefault((t, rel, "r"), []).append(s)
        return True

    def _try_sig(self, sig: str, s: int, pred_pair=None) -> bool:
        st, rel, tt = SIGS[sig]
        probs = LOCALITY.get(sig, DEFAULT_LOCALITY)
        pred = None
        if sig == "T7":
            pred = lambda a, b: self.kind[b] in V.DEVICE_PLATFORM_COMPAT[self.kind[a]]
        t = self._draw(s, tt, probs, pred)
        if t is None:
            return False
        if sig == "T6" and (self.born[s], s) > (self.born[t], t):
            s, t = t, s                                    # evolves_to goes from the earlier to the later CVE
        if sig == "T9" and (t, rel, s) in self.seen:
            return False
        return self._add_edge(s, rel, t, sig)

    def build_edges(self) -> None:
        self.rng = rng = _rng(self.seed, "edges")
        self.seen: set = set()
        self.edges: List[dict] = []
        self.out_adj: Dict[tuple, List[int]] = {}
        self.deg = np.zeros(len(self.nodes), dtype=np.int64)
        self.count = {k: 0 for k in self.p.edge_quotas}
        if self.include_x1:
            self.count["X1"] = 0
        quota = dict(self.p.edge_quotas)

        # (a) coverage: every non-vulnerability node gets one edge in its primary relation
        for t, sig in COVER_SIG.items():
            for i in self.type_idx[t]:
                for _ in range(50):
                    if self.count[sig] >= quota[sig]:
                        raise ValueError(f"quota of {sig} too small to cover all {t} nodes")
                    if self._try_sig(sig, i):
                        break
        # (b) coverage of vulnerabilities through any incoming relation (probability ~ remaining quota)
        for v in self.type_idx["vulnerability"]:
            for _ in range(100):
                if self.deg[v] > 0:
                    break
                rem = np.array([max(quota[k] - self.count[k], 0) for k in V_INCOMING], float)
                if rem.sum() == 0:
                    raise ValueError("incoming-V quotas exhausted before every vulnerability was covered")
                sig = V_INCOMING[int(rng.choice(len(V_INCOMING), p=rem / rem.sum()))]
                st, rel, _ = SIGS[sig]
                src = self._draw(v, st, (0.8, 0.18, 0.02))
                if src is not None:
                    self._add_edge(src, rel, v, sig)
        # (c) fill every relation up to its exact quota
        for sig in CORE_SIGS:
            st = SIGS[sig][0]
            pool = self.pools[(st, "a", 0)]
            attempts = 0
            while self.count[sig] < quota[sig]:
                attempts += 1
                if attempts > 400 * quota[sig] + 10000:
                    raise RuntimeError(f"could not reach quota {quota[sig]} for {sig} (got {self.count[sig]})")
                self._try_sig(sig, pool.draw(rng))
        # (d) optional documented extension X1 = <TA, uses, F>
        if self.include_x1:
            fl = self.type_idx["file"]
            target = 2 * len(fl)
            pool = self.pools[("file", "a", 0)]
            for f in fl + [pool.draw(rng) for _ in range(target - len(fl))]:
                for _ in range(50):
                    a = self._draw(f, "threat_actor", (0.8, 0.18, 0.02))
                    if a is not None and self._add_edge(a, "uses", f, "X1"):
                        break

    # ---------------------------------------------------------------- 4. labels
    def build_labels(self) -> None:
        rng = _rng(self.seed, "labels")
        F, C = self.p.n_families, self.p.n_campaigns
        classes = {t: [f"{CLASS_PREFIX[t]}_c{j + 1:02d}" for j in range(self.p.class_counts[t])] for t in TYPES}
        assign: List[List[str]] = [[] for _ in self.nodes]
        home: Dict[Tuple[str, int], List[int]] = {}
        for t in TYPES:
            K = self.p.class_counts[t]
            fam_dist = []
            for f in range(F):
                h = [c for c in range(K) if c % F == f] or [f % K]
                home[(t, f)] = h
                d = np.full(K, 0.15 / K)
                d[h] += 0.85 * rng.dirichlet(np.full(len(h), 2.0))
                fam_dist.append(d / d.sum())
            camp_dist = []
            for c in range(C):
                d = rng.dirichlet(25 * fam_dist[c % F] + 0.02)
                camp_dist.append(d / d.sum())
            for i in self.type_idx[t]:
                d = camp_dist[self.camp[i]]
                first = int(rng.choice(K, p=d)) if rng.random() > 0.05 else int(rng.integers(K))
                chosen = {first}
                for p_extra in (0.25, 0.05):
                    if K > len(chosen) and rng.random() < p_extra:
                        dd = d.copy()
                        dd[list(chosen)] = 0
                        if dd.sum() > 0:
                            chosen.add(int(rng.choice(K, p=dd / dd.sum())))
                assign[i] = sorted(chosen)
            # minimum support per class: add the class to nodes of the class's home family
            for c in range(K):
                support = sum(1 for i in self.type_idx[t] if c in assign[i])
                if support >= self.p.min_label_support:
                    continue
                f_home = c % F
                pool = [i for i in self.type_idx[t] if self.fam[i] == f_home and c not in assign[i]] or \
                       [i for i in self.type_idx[t] if c not in assign[i]]
                for j in rng.permutation(len(pool))[: self.p.min_label_support - support]:
                    assign[pool[int(j)]] = sorted(assign[pool[int(j)]] + [c])
        labels = {self.nodes[i]["id"]: [classes[self.nodes[i]["type"]][c] for c in assign[i]]
                  for i in range(len(self.nodes))}
        expert = set(int(j) for j in rng.choice(len(self.nodes), size=self.p.n_expert, replace=False))
        source = {self.nodes[i]["id"]: ("expert" if i in expert else "public") for i in range(len(self.nodes))}
        self.label_doc = {
            "labels": labels, "classes": classes, "label_source": source,
            "class_glossary": {c: f"synthetic {t} class {j + 1} (no real-world meaning)"
                               for t in TYPES for j, c in enumerate(classes[t])},
            "n_labelled": len(labels),
        }

    # ---------------------------------------------------------------- 5. attack paths
    def build_paths(self) -> None:
        rng = _rng(self.seed, "paths")
        parsed = {L: [(s, *parse_template(s)) for s in specs] for L, specs in PATH_TEMPLATES.items()}
        by_camp_type: Dict[Tuple[int, str], List[int]] = {}
        for i, n in enumerate(self.nodes):
            by_camp_type.setdefault((self.camp[i], n["type"]), []).append(i)
        plan = [2] * self.p.path_counts[0] + [3] * self.p.path_counts[1] + [4] * self.p.path_counts[2]
        paths, seen = [], set()

        def walk(types, steps, nodes, c):
            k = len(nodes) - 1
            if k == len(steps):
                return list(nodes)
            rel, d = steps[k]
            nxt = list(self.out_adj.get((nodes[-1], rel, d), []))
            for j in rng.permutation(len(nxt)):
                n = nxt[int(j)]
                if self.camp[n] != c or n in nodes or self.nodes[n]["type"] != types[k + 1]:
                    continue
                r = walk(types, steps, nodes + [n], c)
                if r:
                    return r
            return None

        for pid, bucket in enumerate(plan, start=1):
            for attempt in range(5000):
                spec, types, steps = parsed[bucket][int(rng.integers(len(parsed[bucket])))]
                c = int(rng.integers(len(self.camps)))
                starts = by_camp_type.get((c, types[0]), [])
                if not starts:
                    continue
                res = walk(types, steps, [starts[int(rng.integers(len(starts)))]], c)
                if res and tuple(res) not in seen:
                    seen.add(tuple(res))
                    break
            else:
                raise RuntimeError(f"could not instantiate a {bucket}-edge reference path")
            st = []
            for k, (rel, d) in enumerate(steps):
                a, b = (res[k], res[k + 1]) if d == "f" else (res[k + 1], res[k])
                st.append({"source": self.nodes[a]["id"], "relation": rel, "target": self.nodes[b]["id"],
                           "triplet": _SIG_LOOKUP[(self.nodes[a]["type"], rel, self.nodes[b]["type"])],
                           "traversal": "forward" if d == "f" else "reverse"})
            paths.append({"case_id": f"synpath-{pid:03d}", "campaign": self.camps[c]["id"],
                          "family": self.fam_defs[self.camps[c]["family"]]["key"], "template": spec,
                          "n_edges": len(steps), "nodes": [self.nodes[i]["id"] for i in res],
                          "relations": [s["relation"] for s in st], "steps": st,
                          "stage_roles": list(types), **({"provenance": "synthetic"} if self.real else {})})
        self.paths = paths

    # ---------------------------------------------------------------- 6. assemble
    def assemble(self) -> SyntheticDataset:
        members = {c["id"]: {t: [] for t in TYPES} for c in self.camps}
        for n in self.nodes:
            members[n["campaign"]][n["type"]].append(n["id"])
        fam_members: Dict[int, List[str]] = {}
        for c in self.camps:
            fam_members.setdefault(c["family"], []).append(c["id"])
        path_n: Dict[str, int] = {}
        for p in self.paths:
            path_n[p["campaign"]] = path_n.get(p["campaign"], 0) + 1
        camps = []
        for c in self.camps:
            camps.append({
                "campaign_id": c["id"], "codename": c["codename"], "family_id": f"family-{c['family'] + 1:02d}",
                "family": self.fam_defs[c["family"]]["key"], "start": _iso(c["start"]),
                "end": _iso(c["start"] + c["duration"]), "target_countries": c["tgt"], "source_countries": c["src"],
                "industries": c["ind"], "related_campaigns": [x for x in fam_members[c["family"]] if x != c["id"]],
                "n_nodes": sum(len(v) for v in members[c["id"]].values()),
                "nodes_per_type": {t: len(v) for t, v in members[c["id"]].items()},
                "n_reference_paths": path_n.get(c["id"], 0), "members": members[c["id"]]})
        fams = [{"family_id": f"family-{f + 1:02d}", "key": fd["key"], "title": fd["title"],
                 "campaigns": fam_members[f]} for f, fd in enumerate(self.fam_defs)]
        meta = {
            "dataset_type": "synthetic", "purpose": "development_and_pipeline_validation",
            "not_for_reported_experimental_results": True, "seed": self.seed,
            "profile": self.p.name, "generator_version": GENERATOR_VERSION,
            "disclaimer": "Fabricated data. Not derived from AlienVault OTX or any real CTI. Results obtained on this "
                          "dataset must never be presented as reproduction of the manuscript's experiments.",
            "include_x1_extension": self.include_x1,
            "schema": {"entity_types": list(TYPES), "relations": list(RELATIONS),
                       "triplets": {k: list(SIGS[k]) for k in SIGS if k in CORE_SIGS or self.include_x1}},
            "node_counts": {t: len(self.type_idx[t]) for t in TYPES}, "n_nodes": len(self.nodes),
            "n_edges": len(self.edges), "edge_counts_by_triplet": dict(self.count),
            "n_campaigns": len(self.camps), "n_families": len(self.fam_defs),
            "n_reference_paths": len(self.paths), "path_length_buckets": list(self.p.path_counts),
            "n_label_classes": {t: self.p.class_counts[t] for t in TYPES},
            "n_expert_annotated": self.p.n_expert,
            "severity_fields": ["cvss_base_score", "cvss_vector"],
            "manuscript_targets": {"n_nodes": 3728, "n_edges": 11624, "reference_paths": 76,
                                   "path_buckets": {"2_edges": 32, "3_edges": 26, ">=4_edges": 18}},
            "design_choices_not_in_manuscript": [
                "per-triplet edge quotas (manuscript reports only the 11,624 total)",
                "label class names are opaque codes; their semantics are not defined by the manuscript",
                "all nodes are labelled (manuscript text says both '3,614 + 114 = 3,728' and '96.9% labelled')",
                "triplet X1 <TA, uses, F> is NOT generated unless include_x1_extension is true"],
            "id_conventions": {"cve": "CVE-2090..2099-NNNNN (non-existent years)", "domains": ".example/.invalid",
                               "ips": "RFC 5737 documentation ranges", "hashes": "hash of a synthetic string"},
        }
        if self.real:
            prov = Counter(n["provenance"] for n in self.nodes)
            meta.update({
                "dataset_type": "semi_synthetic", "real_component": "public_CVE_metadata",
                "synthetic_components": ["campaigns", "threat_actors", "graph_relationships", "labels",
                                         "reference_attack_paths"],
                "vulnerability_source": "real",
                "disclaimer": "Semi-synthetic: vulnerability nodes are real public CVE records (NVD); all campaigns, "
                              "actors, relations, labels and attack paths are fabricated. No real threat-actor "
                              "attribution is made. Results must never be presented as reproduction of the "
                              "manuscript's experiments.",
                "cve_snapshot": self.snapshot,
                "provenance_counts": {"nodes": dict(prov), "edges": {"synthetic": len(self.edges)},
                                      "attack_paths": {"synthetic": len(self.paths)}},
                "id_conventions": {"cve": "real CVE ids (CVE-YYYY-NNNN+)", "domains": ".example/.invalid",
                                   "ips": "RFC 5737 documentation ranges", "hashes": "hash of a synthetic string"},
            })
            meta["design_choices_not_in_manuscript"] = meta["design_choices_not_in_manuscript"] + [
                "CVE-to-campaign assignment by TF-IDF/SVD clustering of CVE metadata (not in manuscript)",
                "CVE ranking/selection by documented relevance score with a per-vendor cap"]
        edges = sorted(self.edges, key=lambda e: (e["triplet"], e["source"], e["target"]))
        return SyntheticDataset(self.nodes, edges, self.label_doc,
                                {"families": fams, "campaigns": camps}, self.paths, meta)


def _norm(d: Dict[str, float]) -> np.ndarray:
    w = np.array(list(d.values()), float)
    return w / w.sum()


def _resolve_cves(p: Profile, seed: int, pool) -> Tuple[List[dict], dict]:
    if pool is None:
        from .cve_pool import DEFAULT_CACHE
        pool = DEFAULT_CACHE
    if not isinstance(pool, dict):
        pool = load_pool(pool)
    n = p.node_counts["vulnerability"]
    base = select_cves(pool, min(MANUSCRIPT_VULN, len(pool["records"])))
    if n > len(base):
        raise ValueError(f"CVE pool provides {len(base)} selectable CVEs, profile {p.name!r} needs {n}")
    chosen = base
    if n < len(base):                                              # smaller profiles: seeded sample of the selection
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 6]))
        chosen = [base[int(i)] for i in sorted(rng.choice(len(base), size=n, replace=False))]
    snap = {"sha256": pool["sha256"], "source": pool["source"], "retrieved_between": pool["retrieved_between"],
            "pool_format": pool["format"], "n_pool_records": pool["n_records"], "n_selected": n,
            "n_ranked_selection": len(base), "n_queries": len(pool["queries"]),
            "selection_strategy": "relevance score (query tier, ICS vendor/text match, CVSS v3 present, CWE/CPE "
                                  "present) with 6% per-vendor cap; smaller profiles take a seeded sample"}
    return chosen, snap


def generate(profile: str | Profile = "dev", seed: int = 42, include_x1: bool = False,
             vulnerability_source: str = "synthetic", cve_pool=None) -> SyntheticDataset:
    """vulnerability_source: 'synthetic' (fully synthetic, unchanged) or 'real' (semi-synthetic: real NVD CVEs).
    `cve_pool` is a pool dict or a path to cve_pool.json / its directory (default data/raw/cve)."""
    p = PROFILES[profile] if isinstance(profile, str) else profile
    if vulnerability_source not in ("synthetic", "real"):
        raise ValueError("vulnerability_source must be 'synthetic' or 'real'")
    cves = snap = None
    if vulnerability_source == "real":
        cves, snap = _resolve_cves(p, seed, cve_pool)
    b = _Builder(p, seed, include_x1, cves, snap)
    b.build_campaigns()
    b.build_nodes()
    b.build_edges()
    b.build_labels()
    b.build_paths()
    return b.assemble()
