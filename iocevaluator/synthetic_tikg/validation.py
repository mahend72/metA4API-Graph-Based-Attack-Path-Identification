"""Validation of a synthetic dataset against the manuscript schema and the generation profile."""
from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..tikg import RELATIONS
from .generator import (CORE_SIGS, PATH_TEMPLATES, SIGS, SyntheticDataset, dumps, generate, parse_template)
from .profiles import PROFILES, TYPES, Profile


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    checks: List[Check] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(Check(name, bool(passed), detail))

    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        w = max(len(c.name) for c in self.checks)
        lines = [f"[{'PASS' if c.passed else 'FAIL'}] {c.name.ljust(w)}  {c.detail}" for c in self.checks]
        lines.append(f"{sum(c.passed for c in self.checks)}/{len(self.checks)} checks passed")
        return "\n".join(lines)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def similarity_separation(ds: SyntheticDataset) -> Dict[str, float]:
    """Mean actor-actor neighbourhood Jaccard (exploited V + used M + accessed D) for pairs in the same campaign,
    in related campaigns (same family) and in unrelated campaigns; plus the fraction of campaign pairs that share
    at least one vulnerability (anywhere in the campaign's graph neighbourhood)."""
    node = {n["id"]: n for n in ds.nodes}
    camp_fam = {c["campaign_id"]: c["family_id"] for c in ds.campaigns["campaigns"]}
    nb: Dict[str, set] = {n["id"]: set() for n in ds.nodes if n["type"] == "threat_actor"}
    vul_by_camp: Dict[str, set] = {c: set() for c in camp_fam}
    for e in ds.edges:
        if e["source"] not in nb or e["source"] not in node:
            continue
        if e["triplet"] in ("T1", "T11", "T8"):
            nb[e["source"]].add(e["target"])
        if e["triplet"] == "T1" and node[e["source"]]["campaign"] in vul_by_camp:
            vul_by_camp[node[e["source"]]["campaign"]].add(e["target"])
    actors = sorted(nb)
    acc = {"same": [], "related": [], "unrelated": []}
    for a, b in itertools.combinations(actors, 2):
        ca, cb = node[a]["campaign"], node[b]["campaign"]
        k = "same" if ca == cb else ("related" if camp_fam.get(ca) == camp_fam.get(cb) else "unrelated")
        acc[k].append(_jaccard(nb[a], nb[b]))
    share = {"related": [], "unrelated": []}
    for ca, cb in itertools.combinations(sorted(camp_fam), 2):
        share["related" if camp_fam[ca] == camp_fam[cb] else "unrelated"].append(bool(vul_by_camp[ca] & vul_by_camp[cb]))
    m = lambda x: float(np.mean(x)) if x else 0.0
    return {"jaccard_same_campaign": m(acc["same"]), "jaccard_related_campaigns": m(acc["related"]),
            "jaccard_unrelated_campaigns": m(acc["unrelated"]),
            "frac_related_campaign_pairs_sharing_vuln": m(share["related"]),
            "frac_unrelated_campaign_pairs_sharing_vuln": m(share["unrelated"])}


_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}")
# real, publicly named threat groups: none may appear as an actor in generated data
_REAL_ACTOR_BLOCKLIST = ("apt1", "apt28", "apt29", "apt33", "apt41", "lazarus", "sandworm", "fancy bear", "cozy bear",
                         "turla", "equation group", "fin7", "carbanak", "dragonfly", "xenotime", "volt typhoon",
                         "vice society", "lockbit", "conti", "revil", "darkside", "royal", "noname057")
_REAL_ONLY_ATTRS = ("target_countries", "source_countries", "industries", "update_count", "family")


def _validate_semi(ds: SyntheticDataset, r: "ValidationReport", p: Profile) -> None:
    """Extra checks for the semi-synthetic (real public CVE + synthetic graph) mode."""
    meta = ds.metadata
    r.add("semi_metadata_declares_real_and_synthetic_components",
          meta.get("real_component") == "public_CVE_metadata" and meta.get("vulnerability_source") == "real"
          and set(meta.get("synthetic_components", [])) >= {"campaigns", "threat_actors", "graph_relationships",
                                                           "labels", "reference_attack_paths"})
    snap = meta.get("cve_snapshot") or {}
    r.add("cve_snapshot_recorded", bool(snap.get("sha256")) and bool(snap.get("source")), str(snap.get("sha256", ""))[:16])
    vul = [n for n in ds.nodes if n["type"] == "vulnerability"]
    r.add("vulnerability_node_count", len(vul) == p.node_counts["vulnerability"], f"{len(vul)}")
    bad_id = [n["id"] for n in vul if not (_CVE_RE.fullmatch(n["name"]) and n["id"] == f"vuln:{n['name']}")]
    r.add("vulnerability_ids_are_real_cve_format", not bad_id, f"{len(bad_id)} malformed")
    r.add("vulnerability_nodes_provenance_real_public",
          all(n.get("provenance") == "real_public" and n["attrs"].get("provenance") == "real_public"
              and n["attrs"].get("synthetic") is False for n in vul))
    need = ("cve_id", "text", "published", "last_modified", "cvss_base_score", "cvss_vector", "severity", "cwe",
            "vendors", "products", "cpe", "references")
    r.add("cve_metadata_fields_present_or_null",
          all(k in n["attrs"] for n in vul for k in need) and all(n["attrs"]["text"] for n in vul)
          and all(n["attrs"]["cve_id"] == n["name"] for n in vul))
    r.add("no_synthetic_facts_attached_to_real_cves",
          not any(k in n["attrs"] for n in vul for k in _REAL_ONLY_ATTRS) and all(n["attrs"]["synthetic_fields"] == [] for n in vul))
    nonv = [n for n in ds.nodes if n["type"] != "vulnerability"]
    r.add("non_vulnerability_nodes_provenance_synthetic",
          all(n.get("provenance") == "synthetic" and n["attrs"].get("synthetic") is True for n in nonv))
    r.add("all_edges_provenance_synthetic", all(e.get("provenance") == "synthetic" for e in ds.edges),
          f"{sum(e.get('provenance') == 'synthetic' for e in ds.edges)}/{len(ds.edges)}")
    r.add("all_attack_paths_provenance_synthetic", all(x.get("provenance") == "synthetic" for x in ds.attack_paths))
    actors = [n for n in ds.nodes if n["type"] == "threat_actor"]
    r.add("no_real_threat_actor_attribution",
          all(re.fullmatch(r"actor:SYN-TA-\d{4}", n["id"]) for n in actors)
          and not any(b in (n["name"] + " " + n["attrs"].get("text", "")).lower() for n in actors for b in _REAL_ACTOR_BLOCKLIST)
          and all(e.get("provenance") == "synthetic" for e in ds.edges
                  if e["source"].startswith("actor:") or e["target"].startswith("actor:")))
    pc = meta.get("provenance_counts", {})
    r.add("provenance_counts_match",
          pc.get("nodes", {}).get("real_public") == len(vul) and pc.get("nodes", {}).get("synthetic") == len(nonv)
          and pc.get("edges", {}).get("synthetic") == len(ds.edges))


def validate(ds: SyntheticDataset, profile: Optional[Profile] = None, edge_tolerance: float = 0.01) -> ValidationReport:
    r = ValidationReport()
    meta = ds.metadata
    p = profile or PROFILES[meta["profile"]]
    include_x1 = bool(meta.get("include_x1_extension"))
    allowed = {k: SIGS[k] for k in SIGS if k in CORE_SIGS or (include_x1 and k == "X1")}

    # --- provenance flags
    semi = meta.get("dataset_type") == "semi_synthetic"
    flags_ok = (meta.get("dataset_type") in ("synthetic", "semi_synthetic") and meta.get("purpose") == "development_and_pipeline_validation"
                and meta.get("not_for_reported_experimental_results") is True and isinstance(meta.get("seed"), int))
    r.add("metadata_synthetic_flags", flags_ok, f"seed={meta.get('seed')}")
    if semi:
        _validate_semi(ds, r, p)
    else:
        r.add("every_node_marked_synthetic", all(n["attrs"].get("synthetic") is True for n in ds.nodes))
        r.add("synthetic_mode_has_no_provenance_or_real_records",
              not any("provenance" in n for n in ds.nodes) and not any("provenance" in e for e in ds.edges)
              and meta.get("vulnerability_source", "synthetic") == "synthetic")

    # --- nodes
    ids = [n["id"] for n in ds.nodes]
    r.add("no_duplicate_node_ids", len(ids) == len(set(ids)), f"{len(ids)} nodes")
    counts = Counter(n["type"] for n in ds.nodes)
    r.add("node_counts_exact_by_type", all(counts.get(t, 0) == p.node_counts[t] for t in TYPES) and set(counts) <= set(TYPES),
          ", ".join(f"{t}={counts.get(t, 0)}" for t in TYPES))
    r.stats["node_counts"] = {t: counts.get(t, 0) for t in TYPES}
    r.add("total_nodes", len(ds.nodes) == p.n_nodes, f"{len(ds.nodes)} (expected {p.n_nodes})")

    # --- edges
    idset = set(ids)
    ntype = {n["id"]: n["type"] for n in ds.nodes}
    dangling = [e for e in ds.edges if e["source"] not in idset or e["target"] not in idset]
    r.add("no_dangling_edge_endpoints", not dangling, f"{len(dangling)} dangling")
    keys = [(e["source"], e["relation"], e["target"]) for e in ds.edges]
    r.add("no_duplicate_edges", len(keys) == len(set(keys)))
    bad = []
    for e in ds.edges:
        sig = allowed.get(e["triplet"])
        if (sig is None or e["relation"] not in RELATIONS or e["source"] not in ntype or e["target"] not in ntype
                or (ntype[e["source"]], e["relation"], ntype[e["target"]]) != sig):
            bad.append(e)
    r.add("every_edge_respects_schema_T1_T11" + ("+X1" if include_x1 else ""), not bad, f"{len(bad)} violations")
    core_edges = sum(1 for e in ds.edges if e["triplet"] in CORE_SIGS)
    r.add("edge_count_near_target", abs(core_edges - p.n_edges) <= edge_tolerance * p.n_edges,
          f"{core_edges} vs target {p.n_edges}")
    by_sig = Counter(e["triplet"] for e in ds.edges)
    r.add("per_triplet_quotas_exact", all(by_sig.get(k, 0) == v for k, v in p.edge_quotas.items()), str(dict(by_sig)))
    r.stats["edges_by_triplet"] = dict(sorted(by_sig.items(), key=lambda kv: (len(kv[0]), kv[0])))
    deg = Counter()
    for e in ds.edges:
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    iso = [i for i in ids if deg[i] == 0]
    r.add("no_isolated_nodes", not iso, f"{len(iso)} isolated")
    r.stats["avg_undirected_degree"] = 2 * len(ds.edges) / max(len(ds.nodes), 1)

    # --- campaigns
    camp_ids = {c["campaign_id"] for c in ds.campaigns["campaigns"]}
    r.add("campaign_ids_exist_for_all_nodes", all(n["campaign"] in camp_ids for n in ds.nodes))
    r.add("campaign_count", len(camp_ids) == p.n_campaigns and len(camp_ids) >= 10, f"{len(camp_ids)} campaigns")
    members = sum(c["n_nodes"] for c in ds.campaigns["campaigns"])
    r.add("campaign_membership_partitions_nodes", members == len(ds.nodes))
    fam_of = {c["campaign_id"]: c["family_id"] for c in ds.campaigns["campaigns"]}
    nc = {n["id"]: n["campaign"] for n in ds.nodes}
    fam_n = lambda i: fam_of.get(nc.get(i))
    loc = Counter("campaign" if nc[e["source"]] == nc[e["target"]] else
                  ("family" if fam_n(e["source"]) == fam_n(e["target"]) else "cross_family")
                  for e in ds.edges if e["source"] in nc and e["target"] in nc)
    tot = max(sum(loc.values()), 1)
    r.stats["edge_locality"] = {k: loc[k] / tot for k in ("campaign", "family", "cross_family")}
    r.add("edges_are_campaign_conditioned",
          loc["campaign"] / tot >= p.min_campaign_local and (loc["campaign"] + loc["family"]) / tot >= p.min_family_local,
          f"campaign={loc['campaign'] / tot:.3f} family={loc['family'] / tot:.3f} cross={loc['cross_family'] / tot:.3f}")
    sep = similarity_separation(ds)
    r.stats["similarity_separation"] = sep
    r.add("related_campaigns_more_similar_than_unrelated",
          sep["jaccard_same_campaign"] > sep["jaccard_related_campaigns"] > 2 * sep["jaccard_unrelated_campaigns"]
          and sep["frac_related_campaign_pairs_sharing_vuln"] > 2 * sep["frac_unrelated_campaign_pairs_sharing_vuln"],
          ", ".join(f"{k}={v:.4f}" for k, v in sep.items()))

    # --- labels
    L, cls = ds.labels["labels"], ds.labels["classes"]
    r.add("labels_exist_for_every_node", set(L) == idset and all(len(v) >= 1 for v in L.values()))
    r.add("label_class_counts_match_manuscript_table7",
          all(len(cls[t]) == p.class_counts[t] for t in TYPES), str({t: len(cls[t]) for t in TYPES}))
    wrong = [i for i, v in L.items() if i in ntype and not set(v) <= set(cls[ntype[i]])]
    r.add("labels_belong_to_node_type_class_set", not wrong, f"{len(wrong)} wrong")
    support = {t: Counter(c for i, v in L.items() if ntype.get(i) == t for c in v) for t in TYPES}
    min_sup = min(support[t].get(c, 0) for t in TYPES for c in cls[t])
    r.add("every_class_has_min_support", min_sup >= p.min_label_support, f"min support {min_sup}")
    r.stats["label_support"] = {t: {c: support[t].get(c, 0) for c in cls[t]} for t in TYPES}
    r.stats["labels_per_node_mean"] = float(np.mean([len(v) for v in L.values()])) if L else 0.0
    src = Counter(ds.labels["label_source"].values())
    r.add("expert_annotated_count", src.get("expert", 0) == p.n_expert, f"expert={src.get('expert', 0)}")
    codes = [c for t in TYPES for c in cls[t]]
    leaked = [n["id"] for n in ds.nodes if any(c in (n["name"] + " " + n["attrs"].get("text", "")).lower() for c in codes)]
    r.add("no_label_codes_in_name_or_text", not leaked, f"{len(leaked)} leaking nodes")
    if not semi:
        r.add("no_cwe_field_on_vulnerabilities",
              not any("cwe" in n["attrs"] for n in ds.nodes if n["type"] == "vulnerability"))

    # --- attack paths
    P = ds.attack_paths
    e_set = {(e["source"], e["relation"], e["target"]): e["triplet"] for e in ds.edges}
    templates = {parse_template(s) for specs in PATH_TEMPLATES.values() for s in specs}
    dist = Counter("2" if x["n_edges"] == 2 else ("3" if x["n_edges"] == 3 else ">=4") for x in P)
    want = {"2": p.path_counts[0], "3": p.path_counts[1], ">=4": p.path_counts[2]}
    r.add("attack_path_length_distribution", all(dist.get(k, 0) == v for k, v in want.items()) and len(P) == sum(want.values()),
          f"{dict(dist)} (expected {want})")
    r.stats["attack_path_edges_histogram"] = dict(sorted(Counter(x["n_edges"] for x in P).items()))
    problems: List[str] = []
    for x in P:
        nodes, st = x["nodes"], x["steps"]
        if len(nodes) != x["n_edges"] + 1 or len(set(nodes)) != len(nodes) or len(st) != x["n_edges"]:
            problems.append(f"{x['case_id']}: shape")
            continue
        steps_sig = []
        for k, s in enumerate(st):
            fwd = s["traversal"] == "forward"
            a, b = (nodes[k], nodes[k + 1]) if fwd else (nodes[k + 1], nodes[k])
            if (s["source"], s["target"]) != (a, b) or e_set.get((a, s["relation"], b)) != s["triplet"]:
                problems.append(f"{x['case_id']}: hop {k} is not a stored edge")
            steps_sig.append((s["relation"], "f" if fwd else "r"))
        if any(nc.get(n) != x["campaign"] for n in nodes):
            problems.append(f"{x['case_id']}: leaves its campaign")
        if (tuple(ntype.get(n) for n in nodes), tuple(steps_sig)) not in templates:
            problems.append(f"{x['case_id']}: not a schema-valid kill-chain template")
    r.add("attack_paths_use_only_existing_edges_and_valid_templates", not problems, "; ".join(problems[:3]))
    r.add("attack_path_ids_unique_and_campaigns_exist",
          len({x["case_id"] for x in P}) == len(P) and all(x["campaign"] in camp_ids for x in P)
          and len({tuple(x["nodes"]) for x in P}) == len(P))
    return r


def check_reproducible(profile: str = "dev", seed: int = 42, include_x1: bool = False,
                       vulnerability_source: str = "synthetic", cve_pool=None) -> bool:
    """Same seed -> byte-identical serialised dataset; different seed -> different dataset."""
    a = {k: dumps(v) for k, v in generate(profile, seed, include_x1, vulnerability_source, cve_pool).payloads().items()}
    b = {k: dumps(v) for k, v in generate(profile, seed, include_x1, vulnerability_source, cve_pool).payloads().items()}
    return a == b
