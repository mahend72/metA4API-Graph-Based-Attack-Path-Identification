"""The frozen experimental protocol as code: primary settings, the path-search integrity check, sensitivity configurations
and the final-experiment preflight validator.

PROTOCOL_FREEZE.md is the human-readable record; ``FROZEN`` below is the machine-readable one and the preflight checks that
the configuration actually passed to a run equals it.  Nothing here was chosen from a performance number.  Sensitivity
configurations (alpha, tau) are NOT part of the primary experiment: they are reported separately and never selected from.

The real-data feature-exclusion list is deliberately NOT defined here: it depends on the real label definitions, which are
not reconstructed yet (``OPEN_ITEMS``).  The preflight fails until a policy is supplied."""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import pandas as pd

from .path_tracing import PathConfig, path_config_info
from .ranking import RankerConfig

# ------------------------------------------------------------------------------------------------ frozen path search
FROZEN_MAX_EDGES = 6                         # implementation choice (manuscript: ">= 4 edges", no upper bound)
FROZEN_MAX_CANDIDATES = 5000                 # retained post-ranking candidates; cannot change Hit@k (k <= cap) or Edge-F1
UNCAPPED_SAFETY_EXPANSIONS = 5_000_000       # per-seed expansion safety ceiling; must never be hit
UNLIMITED = 10 ** 12                         # "no per-seed path budget"


def exhaustive_path_config(max_edges: int = FROZEN_MAX_EDGES, max_candidates: int = FROZEN_MAX_CANDIDATES) -> PathConfig:
    """The frozen search: no per-seed path budget, a recorded per-seed expansion safety ceiling that must never be hit
    (``TraceResult.search_budget_hit`` is enforced), and a retained-candidate cap that only limits memory.  Every other
    setting is the PathConfig default (confidence 0.5, equal weights for priority / confidence / structure weight,
    coherence 0).  With the ceiling unhit the candidate set is the complete set of schema-valid simple paths between seeds."""
    return PathConfig(max_edges=max_edges, max_paths_per_seed=UNLIMITED, max_candidates=max_candidates,
                      max_expansions_per_seed=UNCAPPED_SAFETY_EXPANSIONS)


class PathIntegrityError(RuntimeError):
    """A path search was cut short by a search budget: the case / run must not produce a result."""


def check_case_integrity(case_id: str, search_budget_hit: bool, n_seeds_path_budget_hit: int = 0,
                         n_seeds_expansion_budget_hit: int = 0) -> None:
    if search_budget_hit:
        raise PathIntegrityError(
            f"integrity validation failed for case {case_id!r}: the path search hit a search budget "
            f"(seeds out of path budget: {n_seeds_path_budget_hit}, seeds out of expansion budget: "
            f"{n_seeds_expansion_budget_hit}); the candidate set is incomplete and no result is produced")


def validate_path_rows(paths: pd.DataFrame) -> None:
    """Post-run check on a ``results_path_cases`` frame: any ``search_budget_hit`` row invalidates the run."""
    if paths is None or len(paths) == 0:
        return
    if "search_budget_hit" not in paths:
        raise PathIntegrityError("path rows carry no search_budget_hit column: the run cannot be validated")
    bad = paths[paths["search_budget_hit"].astype(bool)]
    if len(bad):
        ids = sorted({f"{r.case_id}(fold {r.fold}, seed {r.seed})" for r in bad.itertuples()})[:5]
        raise PathIntegrityError(f"{len(bad)} path case(s) hit a search budget (e.g. {ids}); the run is invalid")


# ------------------------------------------------------------------------------------------------ the frozen protocol
# Every value is also in PROTOCOL_FREEZE.md with its source marker (M manuscript / I implementation choice / D default).
FROZEN: Dict[str, Any] = {
    "status": "frozen",
    "n_splits": 10, "seeds": [0, 1, 2, 3, 4], "fold_seed": 0, "val_fraction": 0.10,
    "ranking": {"alpha": 0.5, "tau": None, "ec_scope": "type", "ec_norm": "max", "missing_severity": "ec_only"},
    "paths": {"conf_threshold": 0.5, "min_edges": 2, "max_edges": FROZEN_MAX_EDGES, "support": "step",
              "max_paths_per_seed": UNLIMITED, "max_expansions_per_seed": UNCAPPED_SAFETY_EXPANSIONS,
              "max_candidates": FROZEN_MAX_CANDIDATES, "priority_top_n": None, "top_k": 5, "match": "strict",
              "weights": {"priority": 1.0, "confidence": 1.0, "structure_weight": 1.0, "coherence": 0.0}},
    "chi_weights": "uniform 1/20 over chi1..chi20",
    "statistics": {"test": "paired Wilcoxon signed-rank on per-fold mean differences", "alpha": 0.05,
                   "correction": "Holm step-down per metric"},
}
OPEN_ITEMS = ("real_data_feature_exclusion_list",)       # the only item allowed to stay open (requires the real label definitions)


# ------------------------------------------------------------------------------------------------ sensitivity configurations
# Fixed a priori on a coarse grid; reported separately, never part of the primary experiment, never selected by performance.
ALPHA_SENSITIVITY = (0.0, 0.25, 0.75, 1.0)
TAU_SENSITIVITY = (0.05, 0.10, 0.25, 0.50)


def primary_ranker_config() -> RankerConfig:
    r = FROZEN["ranking"]
    return RankerConfig(alpha=r["alpha"], tau=r["tau"], ec_scope=r["ec_scope"], ec_norm=r["ec_norm"],
                        missing_severity=r["missing_severity"])


def sensitivity_ranker_configs() -> Dict[str, RankerConfig]:
    """name -> RankerConfig for the alpha and tau sensitivity analyses.  Each varies ONE setting from the primary
    configuration.  ``alpha_tuned_on_validation`` is the manuscript's 'tuned on the validation split' reading
    (``alpha=None``: chosen per fold on validation campaigns only)."""
    base = primary_ranker_config()
    out = {f"alpha_{a:g}": replace(base, alpha=a) for a in ALPHA_SENSITIVITY}
    out["alpha_tuned_on_validation"] = replace(base, alpha=None)
    out.update({f"tau_{t:g}": replace(base, tau=t) for t in TAU_SENSITIVITY})
    return out


def sensitivity_ranker_info() -> Dict[str, Any]:
    return {"role": "sensitivity analysis only; not part of the primary experiment; none is selected by performance",
            "primary": asdict(primary_ranker_config()),
            "configs": {k: {f: v for f, v in asdict(c).items() if getattr(primary_ranker_config(), f) != v}
                        for k, c in sensitivity_ranker_configs().items()}}


# ------------------------------------------------------------------------------------------------ preflight
@dataclass
class FeaturePolicy:
    """The target-defining fields excluded from the features, with where the list comes from.  The real-data list is OPEN:
    it is not invented here and must be supplied once the real label definitions are reconstructed."""
    exclude_fields: Sequence[str]
    strip_terms: Sequence[str] = ()
    source: str = ""
    reasons: Dict[str, str] = field(default_factory=dict)          # field -> why it is excluded (required for every excluded field if given)
    applies_to: Sequence[str] = ()                                  # dataset sources the policy is valid for (empty = unrestricted)
    version: str = ""

    def specified(self) -> bool:
        return bool(self.source.strip()) and (len(self.exclude_fields) > 0 or "none required" in self.source.lower())

    def as_dict(self) -> Dict[str, Any]:
        return {"exclude_fields": list(self.exclude_fields), "strip_terms": list(self.strip_terms), "source": self.source,
                "specified": self.specified(), "applies_to": list(self.applies_to), "version": self.version,
                "reasons": dict(self.reasons),
                "policy_sha256": hashlib.sha256(json.dumps([sorted(self.exclude_fields), sorted(self.strip_terms), sorted(self.reasons.items())],
                                                           sort_keys=True).encode()).hexdigest()}


def load_feature_policy(path: str | Path) -> FeaturePolicy:
    """JSON file ``{"exclude_fields": [...], "strip_terms": [...], "source": "<where the list comes from>"}``.  A policy that
    excludes nothing must say so explicitly: ``"source": "none required: <reason>"``.  Nothing is defaulted or invented."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = sorted(set(d) - {"exclude_fields", "strip_terms", "source", "reasons", "applies_to", "version", "retained_fields", "notes"})
    if unknown:
        raise PreflightError(f"feature policy {path}: unknown keys {unknown}")
    reasons = dict(d.get("reasons", {}))
    if reasons:
        unexplained = sorted(set(d.get("exclude_fields", ())) - set(reasons))
        if unexplained:
            raise PreflightError(f"feature policy {path}: excluded fields without a reason: {unexplained}")
    return FeaturePolicy(tuple(d.get("exclude_fields", ())), tuple(d.get("strip_terms", ())), str(d.get("source", "")),
                         reasons, tuple(d.get("applies_to", ())), str(d.get("version", "")))


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class PreflightReport:
    checks: List[Check] = field(default_factory=list)
    record: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.ok]

    def raise_if_failed(self) -> "PreflightReport":
        if not self.ok:
            raise PreflightError("final-experiment preflight failed:\n" + "\n".join(f"  - {c.name}: {c.detail}" for c in self.failures()))
        return self

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "checks": [asdict(c) for c in self.checks], "record": self.record}


class PreflightError(RuntimeError):
    pass


_STATUS_RE = re.compile(r"^Protocol-status:\s*(\w+)\s*$", re.M)
_OPEN_RE = re.compile(r"^Open-items:\s*(.*)$", re.M)


def read_protocol_file(path: str | Path) -> Dict[str, Any]:
    """The status header of PROTOCOL_FREEZE.md: ``Protocol-status: FROZEN`` and ``Open-items: <comma list | none>``."""
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    text = p.read_text(encoding="utf-8")
    st, op = _STATUS_RE.search(text), _OPEN_RE.search(text)
    items = [] if op is None or op.group(1).strip().lower() in ("", "none") else [x.strip() for x in op.group(1).split(",")]
    return {"exists": True, "status": st.group(1).lower() if st else None, "open_items": items,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def code_sha256(package_dir: Optional[str | Path] = None) -> str:
    root = Path(package_dir) if package_dir else Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _git_state(repo: Optional[str | Path]) -> Dict[str, Any]:
    def run(*a):
        return subprocess.check_output(["git", *a], cwd=repo, stderr=subprocess.DEVNULL, text=True).strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception:
        return {"commit": None, "dirty": None}


def config_sha256(cfg, ranker_cfg: RankerConfig) -> str:
    d = {"eval": asdict(cfg), "ranking": asdict(ranker_cfg)}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def _diff(actual: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    return [f"{k}: {actual.get(k)!r} != frozen {v!r}" for k, v in expected.items() if actual.get(k) != v]


RUN_FINAL, RUN_DIAGNOSTIC, RUN_DRY = "final", "diagnostic", "dry_run"
RUN_CLASSES = (RUN_FINAL, RUN_DIAGNOSTIC, RUN_DRY)


def detect_sensitivity(ranker_cfg: Optional[RankerConfig] = None, paths_cfg: Optional[PathConfig] = None) -> Optional[str]:
    """Name of the sensitivity configuration that ``ranker_cfg`` / ``paths_cfg`` equals, else None.  Sensitivity ranker
    configurations (alpha, tau) and path-budget levels exist for separate, a-priori-fixed analyses and are never primary."""
    if ranker_cfg is not None:
        for name, c in sensitivity_ranker_configs().items():
            if c == ranker_cfg:
                return f"ranker:{name}"
    if paths_cfg is not None:
        from .path_budget import EDGE_LIMITS, LEVELS, level_config      # lazy: path_budget imports this module
        for level, _ in LEVELS:
            for me in sorted({*EDGE_LIMITS, FROZEN_MAX_EDGES}):
                if level_config(level, me) == paths_cfg:
                    return f"paths:{level}/max_edges={me}"
    return None


def _dataset_record(dataset) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"source": getattr(dataset, "source", None), "name": getattr(dataset, "name", None)}
    meta = getattr(dataset, "metadata", None) or {}
    rec["profile"] = meta.get("profile")
    tikg, labels = getattr(dataset, "tikg", None), getattr(dataset, "labels", None)
    if tikg is not None and labels is not None:
        from .evaluation import dataset_fingerprint            # lazy: evaluation imports this module
        rec["n_nodes"] = int(tikg.n)
        rec["fingerprint_sha256"] = dataset_fingerprint(tikg, labels)
    else:
        rec["fingerprint_sha256"] = getattr(dataset, "fingerprint_override", None)      # e.g. 'generator-spec:<hash>' for on-the-fly data
    return rec


def final_experiment_preflight(cfg, ranker_cfg: Optional[RankerConfig] = None, dataset=None, *,
                               feature_policy: Optional[FeaturePolicy] = None,
                               protocol_file: str | Path = "PROTOCOL_FREEZE.md",
                               result=None, allow_dirty: bool = False, repo: Optional[str | Path] = None,
                               n_folds_run: Optional[int] = None, role: str = "primary",
                               run_class: str = RUN_FINAL) -> PreflightReport:
    """Everything that must hold before (``result=None``) and after (``result`` = an EvaluationResult) a final experiment.

    Checks: protocol marked frozen; no unresolved configuration value (only OPEN_ITEMS may remain open); dataset provenance
    known; feature-exclusion policy supplied; folds and seeds fixed; chi1..chi20 weights fixed; ranking / path settings equal
    the frozen protocol; no search budget hit; repository commit, protocol / config / code hashes recorded."""
    rep = PreflightReport()
    add = lambda name, ok, detail="": rep.checks.append(Check(name, bool(ok), "" if ok else detail))   # noqa: E731
    ranker_cfg = ranker_cfg or primary_ranker_config()

    # 0. a final run is the PRIMARY configuration: a sensitivity configuration is never accepted as primary
    sens = detect_sensitivity(ranker_cfg, getattr(cfg, "paths", None))
    add("role_is_primary", role == "primary", f"run role is {role!r}: only the primary configuration may be preflighted as primary")
    add("not_a_sensitivity_configuration", sens is None,
        f"{sens} is a sensitivity configuration (reported separately, never primary, never selected by performance)")
    add("run_class_valid", run_class in RUN_CLASSES, f"run class {run_class!r} not in {RUN_CLASSES}")

    # 1-2. frozen protocol, nothing unresolved
    pf = read_protocol_file(protocol_file)
    add("protocol_marked_frozen", pf.get("exists") and pf.get("status") == "frozen",
        f"{protocol_file}: 'Protocol-status: FROZEN' header missing (found {pf.get('status')!r})")
    extra = sorted(set(pf.get("open_items", [])) - set(OPEN_ITEMS)) if pf.get("exists") else ["<no protocol file>"]
    add("no_unresolved_configuration", not extra and ranker_cfg.alpha is not None,
        f"unresolved items {extra}" if extra else "ranker alpha is None (tuned) - the primary protocol fixes alpha")

    # 3. dataset provenance
    src = getattr(dataset, "source", None)
    add("dataset_provenance_known", src in ("synthetic", "semi_synthetic", "real"),
        f"dataset source {src!r} is not one of synthetic / semi_synthetic / real")
    if src in ("synthetic", "semi_synthetic"):
        rep.record["provenance_note"] = "development / non-manuscript data: results are not reportable as manuscript results"

    # 4. feature-exclusion policy
    add("feature_exclusion_policy_supplied", feature_policy is not None and feature_policy.specified(),
        "no feature-exclusion policy supplied (real-data list is OPEN until the real label definitions are reconstructed)")
    if feature_policy is not None:
        add("feature_policy_in_scope", not feature_policy.applies_to or src in feature_policy.applies_to,
            f"feature policy is only valid for {list(feature_policy.applies_to)}, not for dataset source {src!r} "
            "(the real-data policy is a separate, still OPEN item)")
        add("feature_policy_applied", list(cfg.features.exclude_fields) == list(feature_policy.exclude_fields) and
            list(cfg.features.strip_terms) == list(feature_policy.strip_terms),
            "FeatureConfig exclude_fields / strip_terms differ from the supplied policy")

    # 5. folds and seeds
    add("folds_and_seeds_fixed", not _diff({"n_splits": cfg.n_splits, "seeds": list(cfg.seeds), "fold_seed": cfg.fold_seed,
                                            "val_fraction": cfg.val_fraction},
                                           {k: FROZEN[k] for k in ("n_splits", "seeds", "fold_seed", "val_fraction")})
        and not cfg.max_folds and not cfg.allow_ungrouped,
        "; ".join(_diff({"n_splits": cfg.n_splits, "seeds": list(cfg.seeds), "fold_seed": cfg.fold_seed,
                         "val_fraction": cfg.val_fraction}, {k: FROZEN[k] for k in ("n_splits", "seeds", "fold_seed", "val_fraction")}))
        or "max_folds / allow_ungrouped are smoke-run options and are not allowed in a final run")
    if n_folds_run is not None:
        add("n_splits_used_equals_10", n_folds_run == FROZEN["n_splits"], f"{n_folds_run} folds were run")

    # 6. chi weights
    from .megits import normalise_weights
    from .metagraphs import default_structures
    w = normalise_weights(default_structures())
    add("chi_weights_fixed", len(w) == 20 and all(abs(v - 1 / 20) < 1e-12 for v in w.values()),
        "chi1..chi20 weights are not the uniform 1/20")

    # 7. ranking and path settings
    rk = asdict(ranker_cfg)
    rd = _diff(rk, FROZEN["ranking"])
    add("ranking_matches_frozen", not rd, "; ".join(rd))
    pc = path_config_info(cfg.paths)
    pd_ = _diff(pc, {k: v for k, v in FROZEN["paths"].items()})
    add("paths_match_frozen", not pd_, "; ".join(pd_))
    add("path_integrity_enforced", getattr(cfg, "enforce_path_integrity", False),
        "EvalConfig.enforce_path_integrity is off: a truncated search would silently produce a result")

    # 8. no search budget hit (post-run)
    if result is not None:
        try:
            validate_path_rows(result.paths)
            add("no_search_budget_hit", True)
        except PathIntegrityError as e:
            add("no_search_budget_hit", False, str(e))
    else:
        rep.record["no_search_budget_hit"] = "checked after the run (EvaluationResult) and enforced per case during it"

    # 9. repository / version / config hashes
    g = _git_state(repo)
    add("repository_commit_recorded", g["commit"] is not None, "not a git repository / commit unknown")
    add("clean_working_tree", allow_dirty or g["dirty"] is False,
        "uncommitted changes: commit before the final run (or pass allow_dirty for a dry run)")
    ds_rec = _dataset_record(dataset)
    rep.record.update({"git": g, "protocol_sha256": pf.get("sha256"), "protocol_open_items": pf.get("open_items", []),
                       "config_sha256": config_sha256(cfg, ranker_cfg), "code_sha256": code_sha256(),
                       "frozen": FROZEN, "allow_dirty": bool(allow_dirty),
                       "run_class": run_class, "role": role,
                       "dataset": ds_rec,
                       "feature_policy": feature_policy.as_dict() if feature_policy is not None else None,
                       "folds_seeds": {"n_splits": cfg.n_splits, "seeds": list(cfg.seeds), "fold_seed": cfg.fold_seed,
                                       "val_fraction": cfg.val_fraction, "max_folds": cfg.max_folds,
                                       "n_runs": cfg.n_splits * len(cfg.seeds)},
                       "model_sha256": hashlib.sha256(json.dumps(asdict(cfg.model), sort_keys=True, default=str).encode()).hexdigest()
                       if hasattr(cfg, "model") else None,
                       "ranking": asdict(ranker_cfg), "paths": path_config_info(cfg.paths),
                       "chi_weights": "uniform 1/20", "path_integrity_enforced": bool(getattr(cfg, "enforce_path_integrity", False))})
    add("hashes_recorded", all(rep.record.get(k) for k in ("protocol_sha256", "config_sha256", "code_sha256")),
        "protocol / config / code hash missing")
    rep.record["preflight_checks"] = {c.name: c.ok for c in rep.checks}
    # Reportable as a manuscript result only if EVERYTHING passed, on a clean tree, as a final run, on real data.
    rep.record["reportable_as_manuscript"] = bool(rep.ok and run_class == RUN_FINAL and not allow_dirty
                                                  and g["dirty"] is False and ds_rec["source"] == "real")
    return rep


# ------------------------------------------------------------------------------------------------ preflight summary
def preflight_summary(rep: PreflightReport) -> str:
    """Concise, human-readable block printed BEFORE any training starts (and saved in the manifest as ``preflight``)."""
    r = rep.record
    g = r.get("git", {})
    ds = r.get("dataset", {}) or {}
    fp = r.get("feature_policy") or {}
    fs = r.get("folds_seeds", {}) or {}
    rk, pc = r.get("ranking", {}) or {}, r.get("paths", {}) or {}
    sh = lambda x: (x or "-")[:16]                                                       # noqa: E731
    commit = (g.get("commit") or "unknown")[:12] + ("  DIRTY" if g.get("dirty") else "  clean" if g.get("dirty") is False else "")
    lines = [
        f"PREFLIGHT  run class: {r.get('run_class', '?')}   result: {'PASS' if rep.ok else 'FAIL'}   "
        f"reportable as manuscript result: {'yes' if r.get('reportable_as_manuscript') else 'NO'}",
        f"  protocol hash    : {sh(r.get('protocol_sha256'))}  (open items: {', '.join(r.get('protocol_open_items') or []) or 'none'})",
        f"  code / git       : {commit}   code sha256 {sh(r.get('code_sha256'))}",
        f"  dataset          : {ds.get('source')}/{ds.get('name')}  fingerprint {sh(ds.get('fingerprint_sha256'))}  nodes {ds.get('n_nodes', '?')}",
        f"  feature policy   : {(f"v{fp.get('version') or '?'} {len(fp.get('exclude_fields') or [])} fields excluded, sha {str(fp.get('policy_sha256'))[:12]}; source: " + str(fp.get('source'))[:90]) if fp else 'NOT SUPPLIED'}",
        f"  folds / seeds    : {fs.get('n_splits')} folds x seeds {fs.get('seeds')} = {fs.get('n_runs')} runs per model; "
        f"fold_seed {fs.get('fold_seed')}; val {fs.get('val_fraction')}; max_folds {fs.get('max_folds')}",
        f"  model / config   : model {sh(r.get('model_sha256'))}  config {sh(r.get('config_sha256'))}",
        f"  ranking          : alpha {rk.get('alpha')}  tau {rk.get('tau')}  ec_scope {rk.get('ec_scope')}  "
        f"ec_norm {rk.get('ec_norm')}  missing_severity {rk.get('missing_severity')}",
        f"  paths            : max_edges {pc.get('max_edges')}  conf {pc.get('conf_threshold')}  max_candidates {pc.get('max_candidates')}  "
        f"per-seed budget {pc.get('max_paths_per_seed')}  expansion ceiling {pc.get('max_expansions_per_seed')}  "
        f"integrity enforced: {r.get('path_integrity_enforced')}",
        f"  chi weights      : {r.get('chi_weights', '?')}",
    ]
    bad = rep.failures()
    lines.append(f"  checks           : {len(rep.checks) - len(bad)}/{len(rep.checks)} passed" + ("" if not bad else "; FAILED: " + "; ".join(f"{c.name} ({c.detail})" for c in bad)))
    return "\n".join(lines)


def preflight_manifest_block(rep: PreflightReport) -> Dict[str, Any]:
    """What goes into every manifest: the full preflight record, the per-check outcome and the one-line verdict."""
    return {"performed": True, "ok": rep.ok, "checks": {c.name: c.ok for c in rep.checks},
            "failures": [{"name": c.name, "detail": c.detail} for c in rep.failures()], **rep.record}


# ------------------------------------------------------------------------------------------------ launch session + guard
class PreflightRequiredError(RuntimeError):
    """A manuscript-scale / final-protocol run was started without a passing preflight, or with a configuration that
    differs from the one that was preflighted."""


class ExperimentIntegrityError(PathIntegrityError):
    """The post-run integrity check failed: the run produced no valid result."""


INTENTS = ("development", "final", "primary", "manuscript", "reportable", "full")     # EvalConfig.intent
MANUSCRIPT_SCALE_NODES = 3000          # the manuscript graph has 3,728 nodes
FULL_PROTOCOL_RUNS = 50                # 10 folds x 5 seeds


@dataclass
class ActivePreflight:
    report: PreflightReport
    run_class: str
    protocol_sig: Dict[str, Any]
    ranker_cfg: RankerConfig


_ACTIVE: "contextvars.ContextVar[Optional[ActivePreflight]]" = contextvars.ContextVar("iocevaluator_active_preflight", default=None)


def active_preflight() -> Optional[ActivePreflight]:
    return _ACTIVE.get()


def protocol_signature(cfg) -> Dict[str, Any]:
    """The protocol-critical part of an EvalConfig (the model may legitimately differ per compared variant)."""
    return {"n_splits": cfg.n_splits, "seeds": list(cfg.seeds), "fold_seed": cfg.fold_seed, "val_fraction": cfg.val_fraction,
            "max_folds": cfg.max_folds, "allow_ungrouped": cfg.allow_ungrouped,
            "paths": json.loads(json.dumps(asdict(cfg.paths), sort_keys=True, default=str)),
            "enforce_path_integrity": cfg.enforce_path_integrity,
            "features": json.loads(json.dumps(asdict(cfg.features), sort_keys=True, default=str))}


@contextlib.contextmanager
def preflight_session(report: PreflightReport, cfg, ranker_cfg: Optional[RankerConfig] = None) -> Iterator[ActivePreflight]:
    """Everything run inside is attributed to this preflight: the harness refuses a different configuration and each
    manifest records the preflight.  Refuses to open for a failed preflight, so nothing trains after a failure."""
    report.raise_if_failed()
    act = ActivePreflight(report, report.record.get("run_class", RUN_FINAL), protocol_signature(cfg), ranker_cfg or primary_ranker_config())
    token = _ACTIVE.set(act)
    try:
        yield act
    finally:
        _ACTIVE.reset(token)


def scale_reasons(*, n_nodes: Optional[int], n_splits: Optional[int], n_seeds: Optional[int], max_folds: Optional[int],
                  source: Optional[str] = None) -> List[str]:
    """Why a run counts as final / manuscript-scale (empty -> a development / smoke run)."""
    out = []
    if source == "real":
        out.append("real dataset")
    if n_nodes is not None and n_nodes >= MANUSCRIPT_SCALE_NODES:
        out.append(f"manuscript-scale graph ({n_nodes} nodes >= {MANUSCRIPT_SCALE_NODES})")
    if n_splits and n_seeds and not max_folds and n_splits * n_seeds >= FULL_PROTOCOL_RUNS:
        out.append(f"full-protocol shape ({n_splits} folds x {n_seeds} seeds, no max_folds)")
    return out


def enforce_launch_guard(*, what: str, n_nodes: Optional[int], cfg=None, ranker=None, source: Optional[str] = None,
                         n_splits: Optional[int] = None, seeds: Optional[Sequence[int]] = None,
                         max_folds: Optional[int] = None, require_session: bool = False) -> Optional[ActivePreflight]:
    """The library-level backstop.  A final / manuscript-scale run must run inside a ``preflight_session`` and with the
    preflighted configuration; a development / smoke run may run without one (its manifest then says so)."""
    intent = "development"
    if cfg is not None:
        n_splits, seeds, max_folds = cfg.n_splits, cfg.seeds, cfg.max_folds
        intent = getattr(cfg, "intent", "development")
    if intent not in INTENTS:
        raise PreflightRequiredError(f"{what}: unknown run intent {intent!r}; valid: {INTENTS}")
    reasons = scale_reasons(n_nodes=n_nodes, n_splits=n_splits, n_seeds=len(seeds) if seeds is not None else None,
                            max_folds=max_folds, source=source)
    if intent != "development":                       # explicit marking beats size: a tiny graph marked final is still final
        reasons.append(f"explicitly marked {intent!r}")
    act = _ACTIVE.get()
    if act is None:
        if reasons or require_session:
            raise PreflightRequiredError(
                f"{what}: refused - this is a final / manuscript-scale run ({'; '.join(reasons) or 'session required'}) and no "
                "preflight session is active. Launch it through scripts/run_experiment.py (or wrap the call in "
                "protocol.preflight_session(final_experiment_preflight(...)))")
        return None
    if cfg is not None and protocol_signature(cfg) != act.protocol_sig:
        diff = [k for k, v in protocol_signature(cfg).items() if act.protocol_sig.get(k) != v]
        raise PreflightRequiredError(f"{what}: refused - the configuration differs from the preflighted one ({diff})")
    if ranker is not None:
        rk = getattr(ranker, "cfg", None)
        if rk != act.ranker_cfg:
            raise PreflightRequiredError(
                f"{what}: refused - ranker {getattr(ranker, '__class__', type(ranker)).__name__} "
                f"({'cfg ' + str(rk) if rk is not None else 'no RankerConfig'}) is not the preflighted primary ranker "
                f"({act.ranker_cfg}); sensitivity rankers are never primary")
    return act


def require_preflight_if_marked(what: str, cfg=None, *, protocol_frozen: bool = False) -> None:
    """For the ``run_*_dataset`` convenience runners ("full ablation / model comparison / baseline comparison"): asserting
    ``protocol_frozen=True`` or marking the config final/primary/... is a claim of reportability and needs an active, passing preflight."""
    intent = getattr(cfg, "intent", "development") if cfg is not None else "development"
    if (protocol_frozen or intent != "development") and _ACTIVE.get() is None:
        raise PreflightRequiredError(f"{what}: refused - marked {'protocol_frozen' if protocol_frozen else repr(intent)} (a claim that the run is "
                                     "final / reportable) but no preflight session is active. Use scripts/run_experiment.py")


def attach_preflight(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Add ``manifest['preflight']`` (outside ``content_sha256``: it carries the git commit, which changes between commits)."""
    act = _ACTIVE.get()
    manifest["preflight"] = preflight_manifest_block(act.report) if act is not None else {
        "performed": False, "run_class": "unguarded_development", "reportable": False, "not_reportable": True,
        "note": "no preflight session: a development / smoke run, never reportable as a manuscript result"}
    return manifest


# ------------------------------------------------------------------------------------------------ post-run integrity
def _result_frames(results) -> List[Any]:
    """EvaluationResult | AblationResult | dict of results -> list of objects that have ``.paths`` / ``.manifest``."""
    if results is None:
        return []
    if hasattr(results, "per_variant") and getattr(results, "per_variant"):
        return [results, *results.per_variant.values()]
    if isinstance(results, dict):
        return [x for v in results.values() for x in _result_frames(v)]
    if isinstance(results, (list, tuple)):
        return [x for v in results for x in _result_frames(v)]
    return [results]


# ---- actual chi realisation inside the folds
PRIMARY_CHI_IDS = tuple(range(1, 21))
PRIMARY_CHI_WEIGHT = 1.0 / 20


def realised_fold_record(fold_index: int, builds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One fold's realised adjacency construction(s): the structure ids and weights ``megits_adjacency`` ACTUALLY used."""
    return {"fold": int(fold_index), "n_adjacency_builds": len(builds), "builds": [dict(b) for b in builds]}


def verify_chi_realisation(folds: Sequence[Dict[str, Any]], *, primary_builder: bool = True, expect_folds: Optional[int] = None) -> List[str]:
    """Problems with the structures / weights the folds actually used (empty = chi1..chi20, 1/20 each, in every fold)."""
    out: List[str] = []
    if not primary_builder:
        return out                                   # a custom adjacency (ablation variant) is not the primary configuration
    if expect_folds is not None and len(folds) != expect_folds:
        out.append(f"chi realisation recorded for {len(folds)} folds, expected {expect_folds}")
    for fr in folds:
        f, builds = fr.get("fold"), fr.get("builds") or []
        if not builds:
            out.append(f"fold {f}: no realised chi adjacency was recorded (the adjacency was not built by the frozen MeGiTS builder)")
        for b in builds:
            ids = list(b.get("structure_ids", []))
            missing, extra = sorted(set(PRIMARY_CHI_IDS) - set(ids)), sorted(set(ids) - set(PRIMARY_CHI_IDS))
            if missing:
                out.append(f"fold {f}: chi structure(s) {missing} missing")
            if extra:
                out.append(f"fold {f}: extra structure(s) {extra} present")
            if len(ids) != len(set(ids)):
                out.append(f"fold {f}: duplicated structure ids")
            w = {int(str(k).replace("chi", "")): float(v) for k, v in (b.get("weights") or {}).items()}
            bad = {k: v for k, v in w.items() if abs(v - PRIMARY_CHI_WEIGHT) > 1e-12}
            if bad or set(w) != set(ids):
                out.append(f"fold {f}: weights differ from 1/20: {dict(sorted(bad.items()))}" if bad else f"fold {f}: weight keys differ from structure ids")
            if abs(sum(w.values()) - 1.0) > 1e-9:
                out.append(f"fold {f}: weights sum to {sum(w.values()):.12f}, not 1")
            if b.get("binary"):
                out.append(f"fold {f}: a binary adjacency was built (the primary configuration is weighted MeGiTS)")
    return out


def chi_realisation_block(folds: Sequence[Dict[str, Any]], primary_builder: bool) -> Dict[str, Any]:
    problems = verify_chi_realisation(folds, primary_builder=primary_builder)
    return {"expected": {"structure_ids": list(PRIMARY_CHI_IDS), "weight_each": PRIMARY_CHI_WEIGHT},
            "builder": "megits_default (frozen chi1..chi20)" if primary_builder else "custom adjacency builder (ablation variant; not primary)",
            "primary_builder": bool(primary_builder), "folds": list(folds), "problems": problems, "ok": not problems}


# ---- the final-run integrity layer
def _expected_runs() -> set:
    return {(f, s) for f in range(FROZEN["n_splits"]) for s in FROZEN["seeds"]}


def _frame_integrity(r, label: str) -> List[str]:
    """10 folds x 5 seeds = 50 primary runs: complete, unique, campaign-isolated, path-safe, chi-faithful."""
    fails: List[str] = []
    runs, man = getattr(r, "runs", None), getattr(r, "manifest", None) or {}
    if not isinstance(runs, pd.DataFrame) or not len(runs):
        return [f"{label}: no run rows were produced"]
    exp = _expected_runs()
    if not {"fold", "seed"} <= set(runs.columns):
        return [f"{label}: run table lacks fold / seed columns"]
    keys = list(zip(runs["fold"].astype(int), runs["seed"].astype(int)))
    dups = sorted({k for k in keys if keys.count(k) > 1})
    if dups:
        fails.append(f"{label}: duplicate (fold, seed) rows {dups[:5]}")
    got = set(keys)
    if got - exp:
        fails.append(f"{label}: unexpected (fold, seed) rows {sorted(got - exp)[:5]}")
    if exp - got:
        miss = sorted(exp - got)
        fails.append(f"{label}: missing (fold, seed) rows ({len(miss)}), e.g. {miss[:5]}")
    folds_done = {f for f, _ in got}
    if len(folds_done) != FROZEN["n_splits"]:
        fails.append(f"{label}: {len(folds_done)} outer folds completed, expected {FROZEN['n_splits']}")
    for f in sorted(folds_done):
        seeds = sorted({s for ff, s in got if ff == f})
        if seeds != sorted(FROZEN["seeds"]):
            fails.append(f"{label}: fold {f} completed seeds {seeds}, expected {sorted(FROZEN['seeds'])}")
    if len(runs) != len(exp):
        fails.append(f"{label}: {len(runs)} run rows, expected exactly {len(exp)} primary runs")
    planned = {(d["fold"], d["seed"]) for d in (man.get("protocol", {}).get("runs") or [])}
    if planned and planned != got:
        fails.append(f"{label}: executed runs differ from the manifest's planned runs (a failed / incomplete run was silently omitted)")
    for col in ("macro_f1", "micro_f1"):
        if col in runs.columns and runs[col].isna().any():
            fails.append(f"{label}: {int(runs[col].isna().sum())} runs have undefined {col} (incomplete run)")
    bt = getattr(r, "by_type", None)
    if isinstance(bt, pd.DataFrame) and len(bt) and {"fold", "seed"} <= set(bt.columns):
        if set(zip(bt["fold"].astype(int), bt["seed"].astype(int))) != got:
            fails.append(f"{label}: per-type table does not cover exactly the completed runs")
    tim = getattr(r, "timings", None)
    if isinstance(tim, pd.DataFrame) and len(tim) and len(tim) != len(runs):
        fails.append(f"{label}: timings table has {len(tim)} rows for {len(runs)} runs")
    # campaign isolation from the recorded fold bookkeeping
    finfo = man.get("folds") or []
    if len(finfo) != FROZEN["n_splits"]:
        fails.append(f"{label}: manifest records {len(finfo)} folds, expected {FROZEN['n_splits']}")
    else:
        count: Dict[str, int] = {}
        for fi in finfo:
            tr, va, te = (set(fi.get(k, [])) for k in ("train_campaigns", "val_campaigns", "test_campaigns"))
            for a, b, nm in ((tr, te, "train/test"), (va, te, "validation/test"), (tr, va, "train/validation")):
                if a & b:
                    fails.append(f"{label}: fold {fi.get('fold')}: {nm} campaign overlap {sorted(a & b)[:3]}")
            for c in te:
                count[c] = count.get(c, 0) + 1
        universe = set().union(*(set(finfo[0].get(k, [])) for k in ("train_campaigns", "val_campaigns", "test_campaigns")))
        multi = sorted(c for c, n in count.items() if n != 1)
        if multi:
            fails.append(f"{label}: campaigns in more than one test fold: {multi[:5]}")
        if universe - set(count):
            fails.append(f"{label}: campaigns in no test fold: {sorted(universe - set(count))[:5]}")
    # path diagnostics (only where paths were traced)
    pm = man.get("paths") or {}
    paths = getattr(r, "paths", None)
    if pm.get("enabled") or (isinstance(paths, pd.DataFrame) and len(paths)):
        if not isinstance(paths, pd.DataFrame) or "search_budget_hit" not in paths.columns:
            fails.append(f"{label}: path metrics were produced but search_budget_hit is not exposed")
        bad = _diff(pm, {k: FROZEN["paths"][k] for k in ("max_edges", "conf_threshold", "max_candidates", "max_expansions_per_seed", "max_paths_per_seed")})
        if bad:
            fails.append(f"{label}: path search is not the frozen exhaustive configuration: {bad}")
        if pm.get("enforce_path_integrity") is False:
            fails.append(f"{label}: a run that traced paths did not enforce path integrity")
    # actual chi weights inside the folds (evaluate_cv-produced manifests only; baselines use no chi adjacency)
    if "model" in man or "chi_realisation" in man:
        cr = man.get("chi_realisation")
        if cr is None:
            fails.append(f"{label}: manifest has no chi_realisation record")
        else:
            fails += [f"{label}: {p}" for p in verify_chi_realisation(cr.get("folds", []), primary_builder=cr.get("primary_builder", True),
                                                                    expect_folds=FROZEN["n_splits"] if cr.get("primary_builder", True) else None)]
    return fails


def _hash_view(man: Dict[str, Any]) -> Dict[str, Any]:
    pre = man.get("preflight") or {}
    return {"dataset": (man.get("dataset") or {}).get("fingerprint_sha256"), "protocol": pre.get("protocol_sha256"),
            "config": pre.get("config_sha256"), "code": pre.get("code_sha256"), "commit": (pre.get("git") or {}).get("commit")}


def hash_consistency(manifests: Sequence[Dict[str, Any]], label: str = "outputs") -> List[str]:
    """Every output of one run must carry the same dataset / protocol / config / code hashes (and a performed, passing preflight)."""
    fails: List[str] = []
    views = [_hash_view(m) for m in manifests]
    for key in ("dataset", "protocol", "config", "code", "commit"):
        vals = {v[key] for v in views}
        if None in vals:
            fails.append(f"{label}: {key} hash missing from at least one output")
        elif len(vals) > 1:
            fails.append(f"{label}: inconsistent {key} hash across outputs ({len(vals)} different values)")
    for m in manifests:
        pre = m.get("preflight") or {}
        if not pre.get("performed") or not pre.get("ok"):
            fails.append(f"{label}: an output carries no passing preflight")
            break
    return fails


def post_run_integrity(results, *, expect_folds: Optional[int] = None, raise_on_fail: bool = True, final: bool = False) -> Dict[str, Any]:
    """Re-check ``search_budget_hit`` on every produced path row (and the manifests' enforcement flag / fold count) AFTER
    the run.  Any case that hit the safety ceiling, or any run that did not enforce path integrity, fails the experiment.

    ``final=True`` adds the final-run layer on every produced result: exactly 10 folds x 5 seeds = 50 unique, complete runs;
    campaign isolation; the realised chi1..chi20 weights of every fold; the frozen path diagnostics; and one consistent set of
    dataset / protocol / config / code hashes across all outputs."""
    failures: List[str] = []
    n_rows = 0
    frames = _result_frames(results)
    for r in frames:
        paths = getattr(r, "paths", None)
        if paths is not None and len(paths):
            n_rows += len(paths)
            try:
                validate_path_rows(paths)
            except PathIntegrityError as e:
                failures.append(str(e))
        man = getattr(r, "manifest", None) or {}
        pm = man.get("paths")
        if pm is not None and pm.get("enabled") and pm.get("enforce_path_integrity") is False:
            failures.append("a run that traced paths did not enforce path integrity")
        if expect_folds is not None and man.get("protocol", {}).get("folds_run") not in (None, expect_folds):
            failures.append(f"{man['protocol']['folds_run']} folds were run, expected {expect_folds}")
    if final:
        if not frames:
            failures.append("no results were produced")
        for i, r in enumerate(frames):
            if hasattr(r, "per_variant"):
                continue                                      # the container only aggregates its per-variant results
            man = getattr(r, "manifest", None) or {}
            v = (man.get("variant") or {}).get("name")
            failures += _frame_integrity(r, f"result[{v or i}]")
        failures += hash_consistency([getattr(r, "manifest", None) or {} for r in frames])
    report = {"passed": not failures, "failures": failures, "path_rows_checked": n_rows, "level": "final" if final else "basic",
              "check": "search_budget_hit re-checked after the run" + ("; final-run layer (10x5 completeness, campaign isolation, chi weights, hashes)" if final else "")}
    if failures and raise_on_fail:
        raise ExperimentIntegrityError("post-run integrity check failed: " + " | ".join(failures))
    return report


def verify_outputs(out_dir: str | Path, files: Dict[str, Any], preflight_block: Dict[str, Any], *, n_variants: int = 1) -> List[str]:
    """After writing: every expected output file exists and is non-empty, run tables hold exactly the expected rows, and every
    manifest on disk carries the same dataset / protocol / config / code hashes as the preflight."""
    out, fails = Path(out_dir), []
    if not files:
        return ["no output files were reported by the writer"]
    mans: List[Dict[str, Any]] = []
    for name, p in files.items():
        p = Path(p)
        if not p.exists() or p.stat().st_size == 0:
            fails.append(f"output {name} is missing or empty")
            continue
        if name.endswith("manifest.json"):
            mans.append(json.loads(p.read_text(encoding="utf-8")))
        if name in ("results_per_run.csv", "ablation_runs.csv"):
            n = len(pd.read_csv(p))
            if n != len(_expected_runs()) * n_variants:
                fails.append(f"{name} has {n} rows, expected {len(_expected_runs()) * n_variants}")
    if not mans:
        fails.append("no manifest.json among the outputs")
    pf = out / "preflight.json"
    if pf.exists():
        mans.append({"preflight": json.loads(pf.read_text(encoding="utf-8")), "dataset": (mans[0].get("dataset") if mans else None) or {
            "fingerprint_sha256": (preflight_block.get("dataset") or {}).get("fingerprint_sha256")}})
    else:
        fails.append("preflight.json is missing")
    return fails + hash_consistency(mans, "files on disk")


def reportability(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Whether a manifest may be reported as a manuscript result.  Only a clean-tree FINAL run on real data whose preflight
    passed AND whose final-run integrity layer passed.  Anything produced through a low-level API or a development harness is not."""
    pre, integ, reasons = manifest.get("preflight") or {}, manifest.get("integrity") or {}, []
    if not pre.get("performed"):
        reasons.append("no preflight was performed (development / low-level run)")
    else:
        if not pre.get("ok"):
            reasons.append("preflight did not pass")
        if pre.get("run_class") != RUN_FINAL:
            reasons.append(f"run class is {pre.get('run_class')!r}, not 'final'")
        if (pre.get("git") or {}).get("dirty") is not False:
            reasons.append("the working tree was not clean")
        if not pre.get("reportable_as_manuscript"):
            reasons.append("the data are not real manuscript data")
    if not integ or integ.get("level") != "final" or not integ.get("passed"):
        reasons.append("the final-run integrity layer did not pass")
    return {"reportable": not reasons, "reasons": reasons}


def annotate_integrity(results, integrity: Dict[str, Any]) -> None:
    for r in _result_frames(results):
        man = getattr(r, "manifest", None)
        if isinstance(man, dict):
            man["integrity"] = integrity
            man["reportability"] = reportability(man)


# ------------------------------------------------------------------------------------------------ development-only harnesses
class DevelopmentOnlyError(PreflightRequiredError):
    """A harness that cannot satisfy the frozen protocol was asked to do a final / manuscript-scale run."""


def refuse_at_scale(what: str, *, n_nodes: Optional[int], n_splits: Optional[int] = None, seeds: Optional[Sequence[int]] = None,
                    max_folds: Optional[int] = None, source: Optional[str] = None, remedy: str = "scripts/run_experiment.py") -> None:
    """For harnesses that are NOT the frozen protocol (the legacy ``experiments.run_cv``, the path-budget sensitivity study):
    they stay available for development / smoke runs but refuse anything final or manuscript-scale, even inside a preflight
    session, because their configuration cannot be the preflighted one."""
    reasons = scale_reasons(n_nodes=n_nodes, n_splits=n_splits, n_seeds=len(seeds) if seeds is not None else None,
                            max_folds=max_folds, source=source)
    if reasons:
        raise DevelopmentOnlyError(f"{what}: refused - development-only harness (not the frozen protocol) asked for a "
                                   f"final / manuscript-scale run ({'; '.join(reasons)}). Use {remedy}")


def development_manifest_block(kind: str) -> Dict[str, Any]:
    return {"performed": False, "run_class": kind, "reportable": False, "not_reportable": True,
            "note": "development-only harness: never reportable as a manuscript result"}
