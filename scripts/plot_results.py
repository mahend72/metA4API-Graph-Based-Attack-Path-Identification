"""Reproducible figures from SAVED experiment outputs.  Nothing is computed, tuned, smoothed or filled in here.

    python scripts/plot_results.py                          # scan results/ , write results/figures/
    python scripts/plot_results.py --results-dir results/my_run --out results/figures

Every plotted number is read from files the evaluation framework wrote:
  * means / intervals   -> ``ablation_summary.json`` / ``summary.json`` (``fold_mean`` and ``ci95_folds_t``, the framework's
                           primary interval: 95% Student-t over the per-fold means).  Cost metrics, which the summary does not
                           aggregate, are aggregated from the saved raw timing rows with the framework's own ``aggregate_metric``.
  * significance marks  -> ``ablation_paired.csv`` (``significant_holm_0.05`` from the saved Wilcoxon / Holm results; never recomputed)
  * scalability         -> ``scalability_summary.csv`` (median and the saved q25-q75 range; only sizes that were measured)
A missing file, metric or NaN is shown as "n/a" (or the figure is skipped, with the reason in ``figures_skipped.csv``); it is never
replaced by zero.  Figures for experiments that have not been run are not produced.  One figure per file; PDF + PNG.
``figure_manifest.csv`` links every figure file to its source files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NAN = float("nan")
MANIFEST_COLUMNS = ["figure_filename", "source_result_files", "metrics", "dataset_provenance", "aggregation_method",
                    "uncertainty_method", "config_hash"]
PRIMARY = [("macro_f1", "Macro-F1"), ("micro_f1", "Micro-F1"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]
MODEL_LABELS = {"gcn": "GCN", "gat": "GAT", "hgt": "HGT", "attackg": "AttacKG", "ladder": "LADDER"}
ABLATION_ORDER = [("original_adjacency", "Original adjacency"), ("binary_semantic", "Binary semantic"),
                  ("megits_paths_only", "Meta-paths only"), ("megits_graphs_only", "Meta-graphs only"),
                  ("megits_full", "Full MeGiTS (χ1–χ20)")]
ENTITY_ORDER = ["threat_actor", "vulnerability", "attack_method", "file", "attack_type", "device", "platform"]
ENTITY_LABELS = {"threat_actor": "Threat Actor", "vulnerability": "Vulnerability", "attack_method": "Attack Method",
                 "file": "File", "attack_type": "Attack Type", "device": "Device", "platform": "Platform"}
PATH_BUCKETS = [("", "All reference paths"), ("_2edges", "2-edge"), ("_3edges", "3-edge"), ("_ge4edges", "4+-edge")]
SCAL_COMPONENTS = [("chi_construction", "χ construction"), ("megits_adjacency", "MeGiTS adjacency"), ("train_gcn", "GCN training"),
                   ("train_gat", "GAT training"), ("train_hgt", "HGT training"), ("centrality_ranking", "Ranking"),
                   ("attackg_fit", "AttacKG fit"), ("path_tracing", "Path tracing")]
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]
PROV_LABELS = {"synthetic": "Synthetic data (development/testing only)",
               "semi_synthetic": "Semi-synthetic benchmark (real measured results; not a reproduction of the manuscript's real-data experiments)",
               "real": "Real data"}
UNKNOWN_PROV = "Provenance not recorded in the source files"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100, "savefig.bbox": "tight"})


# ------------------------------------------------------------------------------------------------ reading
def _num(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    return v if math.isfinite(v) else NAN


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None


@dataclass
class Stat:
    """A value with its saved uncertainty.  Anything missing is NaN (never 0)."""
    value: float = NAN
    lo: float = NAN
    hi: float = NAN
    n: int = 0

    @property
    def ok(self) -> bool:
        return math.isfinite(self.value)

    @property
    def err(self) -> Optional[Tuple[float, float]]:
        if self.ok and math.isfinite(self.lo) and math.isfinite(self.hi):
            return max(self.value - self.lo, 0.0), max(self.hi - self.value, 0.0)
        return None


def stat_from_aggregate(agg: Optional[dict]) -> Stat:
    """``aggregate_metric`` output -> fold mean with the primary interval ``ci95_folds_t``."""
    if not isinstance(agg, dict):
        return Stat()
    ci = agg.get("ci95_folds_t") or [NAN, NAN]
    return Stat(_num(agg.get("fold_mean")), _num(ci[0]), _num(ci[1]), int(agg.get("n_folds") or 0))


def provenance_label(manifest: Optional[dict]) -> str:
    p = (manifest or {}).get("provenance")
    src = p.get("source") if isinstance(p, dict) else None
    if src is None:
        src = ((manifest or {}).get("dataset") or {}).get("source") if isinstance((manifest or {}).get("dataset"), dict) else None
    if src not in PROV_LABELS:
        return UNKNOWN_PROV
    label = PROV_LABELS[src]
    if src == "real" and isinstance(p, dict) and not p.get("protocol_frozen"):
        label += " — protocol not frozen"
    return label


class ResultDir:
    """One experiment output directory (anything with a ``manifest.json``)."""

    def __init__(self, path: Path, manifest: dict):
        self.path, self.manifest = Path(path), manifest
        kind = manifest.get("kind")
        if kind not in ("ablation_study", "model_comparison", "external_baseline_comparison", "scalability"):
            kind = "evaluation" if (self.path / "summary.json").exists() else (kind or "unknown")
        self.kind = kind
        self.prov = provenance_label(manifest)
        self.prov_known = self.prov != UNKNOWN_PROV
        self.hash = str(manifest.get("content_sha256") or "")
        self.tag = re.sub(r"[^A-Za-z0-9._-]+", "_", self.path.name) or "results"

    def file(self, name: str) -> Path:
        return self.path / name

    def summaries(self) -> Tuple[Dict[str, dict], Optional[str], List[Path]]:
        """variant -> summary (``overall`` / ``by_entity_type``), the reference variant, and the file read."""
        d = read_json(self.file("ablation_summary.json"))
        if d and isinstance(d.get("variants"), dict):
            return d["variants"], d.get("reference_variant"), [self.file("ablation_summary.json")]
        d = read_json(self.file("summary.json"))
        if d:
            m = self.manifest.get("model")
            name = m.get("model") if isinstance(m, dict) and m.get("model") else "model"
            return {name: d}, name, [self.file("summary.json")]
        return {}, None, []

    def variant_meta(self) -> Dict[str, dict]:
        return {v.get("name"): v for v in self.manifest.get("variants", []) if isinstance(v, dict)}


def _dry_path(path: Path, root: Path) -> bool:
    try:
        parts = Path(path).resolve().relative_to(Path(root).resolve()).parts
    except ValueError:
        parts = Path(path).parts[-1:]
    return any("dry" in p.lower() for p in parts)


def is_dry_run(rd: "ResultDir", root: Path) -> bool:
    """Dry-run outputs (preflight run class ``dry_run``, or any directory below the results root named *dry*) are plumbing checks,
    not experiments."""
    pf = rd.manifest.get("preflight")
    cls = pf.get("run_class") if isinstance(pf, dict) else None
    return cls == "dry_run" or _dry_path(rd.path, root)


def discover(root: Path) -> List[ResultDir]:
    out = []
    for mf in sorted(Path(root).rglob("manifest.json")):
        if "figures" in mf.parts:
            continue
        m = read_json(mf)
        if isinstance(m, dict):
            out.append(ResultDir(mf.parent, m))
    return out


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return Path(p).as_posix()


# ------------------------------------------------------------------------------------------------ output bookkeeping
class Figures:
    def __init__(self, out: Path, dpi: int = 300, formats: Sequence[str] = ("pdf", "png")):
        self.out, self.dpi, self.formats = Path(out), dpi, tuple(formats)
        self.rows: List[Dict[str, str]] = []
        self.skipped: List[Dict[str, str]] = []
        self.figs: Dict[str, plt.Figure] = {}

    def remove_stale(self) -> None:
        """Delete figure files from an earlier run of this script (``figNN_*.pdf/png``) so the folder always matches the manifest."""
        for f in self.out.glob("fig[0-9]*__*.*"):
            if f.suffix in (".pdf", ".png"):
                f.unlink()

    def skip(self, figure: str, reason: str) -> None:
        self.skipped.append({"figure": figure, "reason": reason})
        print(f"[skip] {figure}: {reason}")

    def add(self, name: str, fig: plt.Figure, sources: Sequence[Path], metrics: str, prov: str, aggregation: str,
            uncertainty: str, cfg_hash: str = "") -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        if not cfg_hash:                                    # no experiment hash recorded: identify the exact source bytes instead
            import hashlib
            h = hashlib.sha256()
            for sp in dict.fromkeys(Path(x) for x in sources):
                h.update(Path(sp).read_bytes() if Path(sp).exists() else b"")
            cfg_hash = "source-files-sha256:" + h.hexdigest()
        fig.text(0.5, -0.07, prov, ha="center", va="top", fontsize=8, style="italic", wrap=True)
        for ext in self.formats:
            fn = f"{name}.{ext}"
            fig.savefig(self.out / fn, dpi=self.dpi)
            self.rows.append({"figure_filename": fn, "source_result_files": ";".join(dict.fromkeys(_rel(s) for s in sources)),
                              "metrics": metrics, "dataset_provenance": prov, "aggregation_method": aggregation,
                              "uncertainty_method": uncertainty, "config_hash": cfg_hash})
        self.figs[name] = fig
        print(f"[ok]   {name}")

    def close(self) -> None:
        for f in self.figs.values():
            plt.close(f)
        self.figs = {}

    def write_manifests(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.out / "figure_manifest.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
            w.writeheader()
            w.writerows(self.rows)
        with open(self.out / "figures_skipped.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["figure", "reason"])
            w.writeheader()
            w.writerows(self.skipped)


UNC = "95% Student-t interval over per-fold means (ci95_folds_t) as saved by the evaluation framework"
AGG = "fold_mean: mean over folds of the seed-averaged fold values, as saved in the summary JSON"


def _title(ax, title: str, prov_short: str) -> None:
    ax.set_title(f"{title}\n{prov_short}", loc="left")


def short_prov(label: str) -> str:
    return "Semi-synthetic benchmark" if label.startswith("Semi-synthetic") else \
        "Synthetic data (development only)" if label.startswith("Synthetic") else \
        "Real data" if label.startswith("Real") else "Provenance not recorded"


def grouped_bars(groups: Sequence[str], series: Sequence[Tuple[str, Sequence[Stat]]], ylabel: str, title: str, prov: str,
                 horizontal: bool = False, log: bool = False, ylim: Optional[Tuple[float, float]] = (0.0, 1.05)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(6.5, 1.1 * len(groups) * max(len(series), 2) * 0.55 + 2), 4.6))
    n = max(len(series), 1)
    w = 0.8 / n
    x = np.arange(len(groups))
    for i, (lab, stats) in enumerate(series):
        pos = x - 0.4 + w * (i + 0.5)
        vals = [s.value if s.ok else NAN for s in stats]
        errs = [s.err for s in stats]
        yerr = np.array([[e[0] if e else 0.0 for e in errs], [e[1] if e else 0.0 for e in errs]])
        if log:                                            # bars on a log axis would start at an arbitrary baseline: use points
            okm = np.isfinite(vals)
            ax.errorbar(pos[okm], np.array(vals)[okm], yerr=yerr[:, okm], fmt="o", capsize=3, color=COLORS[i % len(COLORS)], label=lab)
        else:
            ax.bar(pos, vals, w * 0.92, label=lab, color=COLORS[i % len(COLORS)],
                   yerr=yerr, capsize=2.5, error_kw={"elinewidth": 1.0, "ecolor": "#333333"})
        for p, s in zip(pos, stats):                       # a missing value is annotated, not drawn as a zero bar
            if not s.ok and not log:
                ax.annotate("n/a", (p, 0), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", rotation=90,
                            fontsize=8, color="#666666")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=0 if len(groups) <= 5 else 30, ha="center" if len(groups) <= 5 else "right")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    elif ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=min(len(series), 5), loc="upper center", bbox_to_anchor=(0.5, -0.12 if len(groups) <= 5 else -0.28))
    _title(ax, title, short_prov(prov))
    return fig


def get(summ: Dict[str, dict], variant: str, metric: str, etype: Optional[str] = None) -> Stat:
    s = summ.get(variant) or {}
    if etype is None:
        return stat_from_aggregate((s.get("overall") or {}).get(metric))
    return stat_from_aggregate(((s.get("by_entity_type") or {}).get(etype) or {}).get(metric))


def _any(stats: Sequence[Stat]) -> bool:
    return any(s.ok for s in stats)


# ------------------------------------------------------------------------------------------------ 1 model comparison
def fig_model_comparison(rd: ResultDir, F: Figures) -> None:
    name = f"fig01_model_comparison__{rd.tag}"
    summ, _, srcs = rd.summaries()
    models = [m for m in MODEL_LABELS if m in summ]
    series = [(MODEL_LABELS[m], [get(summ, m, k) for k, _ in PRIMARY]) for m in models]
    if len([s for s in series if _any(s[1])]) < 2:
        return F.skip(name, f"fewer than two models with saved metrics in {_rel(rd.path)}")
    missing = [MODEL_LABELS[m] for m in MODEL_LABELS if m not in summ]
    fig = grouped_bars([l for _, l in PRIMARY], series, "Score (mean ± 95% CI over folds)", "Model comparison", rd.prov)
    if missing:
        fig.axes[0].text(0.99, 0.02, "not in source: " + ", ".join(missing), transform=fig.axes[0].transAxes, ha="right", fontsize=8,
                         color="#666666")
    F.add(name, fig, srcs, ",".join(k for k, _ in PRIMARY), rd.prov, AGG, UNC, rd.hash)


# ------------------------------------------------------------------------------------------------ 2 ablation
def fig_ablation(rd: ResultDir, F: Figures) -> None:
    name = f"fig02_ablation__{rd.tag}"
    summ, _, srcs = rd.summaries()
    present = [(v, l) for v, l in ABLATION_ORDER if v in summ]
    if len(present) < 2:
        return F.skip(name, f"fewer than two adjacency-ablation variants saved in {_rel(rd.path)}")
    series = [(l, [get(summ, v, k) for k, _ in PRIMARY]) for v, l in present]
    absent = [l for v, l in ABLATION_ORDER if v not in summ]
    fig = grouped_bars([l for _, l in PRIMARY], series, "Score (mean ± 95% CI over folds)", "Adjacency ablation", rd.prov)
    if absent:
        fig.axes[0].text(0.99, 0.02, "not in source: " + ", ".join(absent), transform=fig.axes[0].transAxes, ha="right", fontsize=8,
                         color="#666666")
    F.add(name, fig, srcs, ",".join(k for k, _ in PRIMARY), rd.prov, AGG, UNC, rd.hash)


# ------------------------------------------------------------------------------------------------ 3 individual chi
def fig_chi(rd: ResultDir, F: Figures, metric: str = "macro_f1", label: str = "Macro-F1") -> None:
    name = f"fig03_chi_contribution_{metric}__{rd.tag}"
    summ, ref, srcs = rd.summaries()
    ref = ref if ref in summ else "megits_full"
    rows = [(k, get(summ, f"chi{k}", metric)) for k in range(1, 21) if f"chi{k}" in summ]
    rows = [(k, s) for k, s in rows if s.ok]
    if not rows:
        return F.skip(name, f"no single-χ variants with saved {metric} in {_rel(rd.path)}")
    rows.sort(key=lambda t: t[1].value)
    paired = read_csv(rd.file("ablation_paired.csv"))
    sig: Dict[str, bool] = {}
    if paired is not None and {"variant", "metric", "significant_holm_0.05"} <= set(paired.columns):
        for _, r in paired[paired.metric == metric].iterrows():
            if math.isfinite(_num(r.get("wilcoxon_p_holm"))):   # a test that could not be run is not a "non-significant" one
                sig[r["variant"]] = bool(r["significant_holm_0.05"])
        srcs = srcs + [rd.file("ablation_paired.csv")]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 2.2))
    y = np.arange(len(rows))
    vals = [s.value for _, s in rows]
    xerr = np.array([[s.err[0] if s.err else 0.0 for _, s in rows], [s.err[1] if s.err else 0.0 for _, s in rows]])
    ax.barh(y, vals, color=COLORS[0], xerr=xerr, capsize=2, error_kw={"elinewidth": 1, "ecolor": "#333333"})
    for i, (k, s) in enumerate(rows):
        if sig.get(f"chi{k}"):
            ax.annotate("*", (s.hi if math.isfinite(s.hi) else s.value, i), xytext=(4, -3), textcoords="offset points", fontsize=14)
    ax.set_yticks(y)
    ax.set_yticklabels([f"χ{k}" for k, _ in rows])
    refstat = get(summ, ref, metric)
    if refstat.ok:
        ax.axvline(refstat.value, color=COLORS[4], ls="--", lw=1.4, label=f"Full MeGiTS χ1–χ20: {refstat.value:.3f}")
        if refstat.err:
            ax.axvspan(refstat.lo, refstat.hi, color=COLORS[4], alpha=0.12)
    ax.set_xlabel(f"{label} (mean ± 95% CI over folds)")
    note = "* paired Wilcoxon vs full model, Holm-adjusted p < 0.05 (saved ablation_paired.csv)" if sig else \
        "significance: not available in the saved results"
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    _title(ax, f"Single-χ contribution vs full model ({label})", short_prov(rd.prov) + "\n" + note)
    F.add(name, fig, srcs, metric, rd.prov, AGG, UNC + "; significance marks from saved Wilcoxon/Holm table", rd.hash)


# ------------------------------------------------------------------------------------------------ 4/5 label fraction, depth
def _line_figure(rd: ResultDir, F: Figures, name: str, title: str, xlabel: str, points: List[Tuple[float, str]],
                 summ: Dict[str, dict], srcs: List[Path]) -> None:
    points = sorted(points)
    if len(points) < 2:
        return F.skip(name, f"fewer than two settings saved in {_rel(rd.path)} (found {len(points)})")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, (k, lab) in enumerate(PRIMARY):
        st = [get(summ, v, k) for _, v in points]
        xs = np.array([x for x, _ in points], dtype=float)
        ys = np.array([s.value for s in st])
        ok = np.isfinite(ys)
        if not ok.any():
            continue
        err = np.array([s.err if s.err else (0.0, 0.0) for s in st]).T
        ax.errorbar(xs[ok], ys[ok], yerr=err[:, ok], marker="o", capsize=3, color=COLORS[i], label=lab, lw=1.5)   # NaN points are left out
    ax.set_xticks([x for x, _ in points])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score (mean ± 95% CI over folds)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    _title(ax, title, short_prov(rd.prov))
    F.add(name, fig, srcs, ",".join(k for k, _ in PRIMARY), rd.prov, AGG, UNC, rd.hash)


def _setting(rd: ResultDir, variant: str, key: str, pattern: str) -> Optional[float]:
    """The setting of a variant: the saved manifest value (preferred), else the number in the variant's name."""
    v = rd.variant_meta().get(variant, {})
    if math.isfinite(_num(v.get(key))):
        return _num(v[key])
    m = re.search(pattern, variant)
    return float(m.group(1)) if m else None


def fig_label_fraction(rd: ResultDir, F: Figures) -> None:
    summ, ref, srcs = rd.summaries()
    pts = []
    for v in summ:
        if v == "megits_full":                                  # the reference variant is trained on all labels by definition
            pts.append((100.0, v))
        elif re.fullmatch(r"megits_full_lf\d+", v):
            frac = rd.variant_meta().get(v, {}).get("label_fraction")
            x = _num(frac) * 100.0 if math.isfinite(_num(frac)) else _setting(rd, v, "label_fraction", r"_lf(\d+)$")
            if x is not None and math.isfinite(x):
                pts.append((x, v))
    _line_figure(rd, F, f"fig04_label_fraction__{rd.tag}", "Reduced-label experiment", "Labelled training data (%)", pts, summ, srcs)


def fig_depth(rd: ResultDir, F: Figures) -> None:
    summ, ref, srcs = rd.summaries()
    pts = []
    for v in summ:
        if v == "megits_full":                                  # the reference variant is the 2-layer GCN by definition
            pts.append((_setting(rd, v, "layers", r"$^") or 2.0, v))
        elif re.fullmatch(r"megits_full_\d+layer", v):
            x = _setting(rd, v, "layers", r"_(\d+)layer$")
            if x is not None:
                pts.append((x, v))
    _line_figure(rd, F, f"fig05_gcn_depth__{rd.tag}", "GCN depth sensitivity", "Number of GCN layers", pts, summ, srcs)


# ------------------------------------------------------------------------------------------------ 6 per entity type
def fig_per_type(rd: ResultDir, F: Figures, metric: str = "macro_f1", label: str = "Macro-F1") -> None:
    name = f"fig06_per_entity_type_{metric}__{rd.tag}"
    summ, ref, srcs = rd.summaries()
    cand = [m for m in MODEL_LABELS if m in summ] if rd.kind in ("model_comparison", "external_baseline_comparison") else \
        [ref] if ref in summ else []
    types = [t for t in ENTITY_ORDER if any(t in (summ[v].get("by_entity_type") or {}) for v in cand)]
    series = [((MODEL_LABELS.get(v, "Full MeGiTS (χ1–χ20)" if v == "megits_full" else v)), [get(summ, v, metric, t) for t in types])
              for v in cand]
    series = [s for s in series if _any(s[1])]
    if not types or not series:
        return F.skip(name, f"no per-entity-type {metric} in {_rel(rd.path)}")
    fig = grouped_bars([ENTITY_LABELS[t] for t in types], series, f"{label} (mean ± 95% CI over folds)", f"Per-entity-type {label}", rd.prov)
    F.add(name, fig, srcs, metric, rd.prov, AGG + " (by_entity_type)", UNC, rd.hash)


# ------------------------------------------------------------------------------------------------ 7 attack paths
def fig_paths(rd: ResultDir, F: Figures) -> None:
    name = f"fig07_attack_path__{rd.tag}"
    summ, ref, srcs = rd.summaries()
    var = ref if ref in summ else (next(iter(summ)) if summ else None)
    if var is None:
        return F.skip(name, f"no summaries in {_rel(rd.path)}")
    mets = [("exact_path_hit5", "Exact-Path Hit@5"), ("path_edge_f1", "Edge-F1"), ("path_relation_f1", "Relation-F1")]
    buckets = [(suf, lab) for suf, lab in PATH_BUCKETS if any(get(summ, var, m + suf).ok for m, _ in mets)]
    if not buckets:
        return F.skip(name, f"no attack-path metrics saved for {var} in {_rel(rd.path)} (run with reference paths)")
    series = [(lab, [get(summ, var, m + suf) for suf, _ in buckets]) for m, lab in mets]
    series = [s for s in series if _any(s[1])]
    fig = grouped_bars([l for _, l in buckets], series, "Score (mean ± 95% CI over folds)",
                       f"Attack-path evaluation ({MODEL_LABELS.get(var, var)})", rd.prov)
    F.add(name, fig, srcs, ",".join(m + suf for m, _ in mets for suf, _ in buckets), rd.prov, AGG, UNC, rd.hash)


# ------------------------------------------------------------------------------------------------ 8 prioritisation
def fig_topk(rd: ResultDir, F: Figures) -> None:
    name = f"fig08a_prioritisation_topk__{rd.tag}"
    summ, _, srcs = rd.summaries()
    models = [m for m in MODEL_LABELS if m in summ]
    ks = [(k, f"Top-{k[3:-9]} hit rate") for k in ("top5_hit_rate", "top10_hit_rate")]
    series = [(lab, [get(summ, m, k) for m in models]) for k, lab in ks]
    series = [s for s in series if _any(s[1])]
    if not series:
        return F.skip(name, f"no top-k hit rate saved in {_rel(rd.path)} (no top-k cases were supplied)")
    fig = grouped_bars([MODEL_LABELS[m] for m in models], series, "Hit rate (mean ± 95% CI over folds)",
                       "Threat prioritisation: top-k hit rate (fused ranking)", rd.prov)
    F.add(name, fig, srcs, "top5_hit_rate,top10_hit_rate", rd.prov, AGG, UNC, rd.hash)


def fig_ranking_distribution(csv_path: Path, F: Figures) -> None:
    name = f"fig08b_ranking_distribution__{re.sub(r'[^A-Za-z0-9._-]+', '_', csv_path.parent.name)}"
    df = read_csv(csv_path)
    if df is None or not {"eigenvector_centrality", "risk"} <= set(df.columns):
        return F.skip(name, f"{_rel(csv_path)} lacks eigenvector_centrality / risk")
    df = df.dropna(subset=["eigenvector_centrality", "risk"])
    if df.empty:
        return F.skip(name, f"{_rel(csv_path)} has no finite rows")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for i, (t, g) in enumerate(df.groupby("type") if "type" in df else [("IOC", df)]):
        ax.scatter(g.eigenvector_centrality, g.risk, s=18, color=COLORS[i % len(COLORS)], label=ENTITY_LABELS.get(t, t), alpha=0.85)
    ax.set_xlabel("Eigenvector centrality")
    ax.set_ylabel("Fused risk R = α·EC + (1−α)·Severity")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    prov = UNKNOWN_PROV + "; output of a ranking run (top-N list only), not an evaluation result"
    ax.set_title(f"Top-{len(df)} ranked IOCs (not the full population)\nRanking-run output", loc="left")
    F.add(name, fig, [csv_path], "eigenvector_centrality,risk", prov, "none: saved per-IOC values", "none (no aggregation)")


def fig_sensitivity(dirs: List[ResultDir], F: Figures) -> None:
    groups: Dict[Path, List[ResultDir]] = {}
    for rd in dirs:
        if re.match(r"^(alpha|tau)_", rd.path.name):
            groups.setdefault(rd.path.parent, []).append(rd)
    for parent, members in groups.items():
        for param in ("alpha", "tau"):
            name = f"fig08c_sensitivity_{param}__{re.sub(r'[^A-Za-z0-9._-]+', '_', parent.name)}"
            pts, cats = [], []
            for rd in members:
                if not rd.path.name.startswith(param + "_"):
                    continue
                suff = rd.path.name[len(param) + 1:]
                (pts if re.fullmatch(r"[0-9.eE+-]+", suff) else cats).append((suff, rd))
            pts.sort(key=lambda t: float(t[0]))
            allp = pts + sorted(cats)
            if len(allp) < 2:
                if allp:
                    F.skip(name, f"fewer than two {param} settings saved under {_rel(parent)}")
                continue
            fig, ax = plt.subplots(figsize=(6.4, 4.4))
            srcs = []
            drawn = False
            for i, (k, lab) in enumerate([("top5_hit_rate", "Top-5 hit rate"), ("top10_hit_rate", "Top-10 hit rate")]):
                st = []
                for _, rd in allp:
                    s, ref, f = rd.summaries()
                    srcs += f
                    st.append(get(s, ref, k) if ref else Stat())
                xs = np.arange(len(allp), dtype=float)
                ys = np.array([s.value for s in st])
                ok = np.isfinite(ys)
                if ok.any():
                    err = np.array([s.err if s.err else (0.0, 0.0) for s in st]).T
                    ax.errorbar(xs[ok], ys[ok], yerr=err[:, ok], marker="o", capsize=3, color=COLORS[i], label=lab)
                    drawn = True
            if not drawn:
                plt.close(fig)
                F.skip(name, f"no top-k hit rate saved for the {param} settings under {_rel(parent)}")
                continue
            ax.set_xticks(np.arange(len(allp)))
            ax.set_xticklabels([s.replace("_", " ") for s, _ in allp], rotation=20)
            ax.set_xlabel(f"{'α' if param == 'alpha' else 'τ'} setting")
            ax.set_ylabel("Hit rate (mean ± 95% CI over folds)")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            ax.legend(frameon=False)
            prov = allp[0][1].prov
            ax.set_title(f"SENSITIVITY ANALYSIS — not the primary result\n{short_prov(prov)}", loc="left", color="#B00020")
            F.add(name, fig, srcs, "top5_hit_rate,top10_hit_rate", prov, AGG + "; settings named by the saved config directories", UNC,
                  allp[0][1].hash)


# ------------------------------------------------------------------------------------------------ 9 computational cost
def fig_cost(rd: ResultDir, F: Figures) -> None:
    from iocevaluator.evaluation import aggregate_metric          # the framework's own fold aggregation
    tim, runs = read_csv(rd.file("ablation_timings.csv")), read_csv(rd.file("ablation_runs.csv"))
    summ, _, _ = rd.summaries()
    models = [m for m in MODEL_LABELS if (tim is not None and m in set(tim.get("variant", [])))
              or (runs is not None and m in set(runs.get("variant", [])))]
    specs = [("train", "train_time_total_s", tim, "Training time per run (s)", True),
             ("inference", "inference_time_s", tim, "Inference time per run (s)", True),
             ("parameters", "n_parameters", runs, "Trainable parameters", True)]
    for key, col, df, ylabel, log in specs:
        name = f"fig09_{key}__{rd.tag}"
        if df is None or col not in df or "variant" not in df:
            F.skip(name, f"{col} not saved in {_rel(rd.path)}")
            continue
        stats = []
        for m in models:
            sub = df[df.variant == m]
            stats.append(stat_from_aggregate(aggregate_metric(sub, col)) if len(sub) and sub[col].notna().any() else Stat())
        if sum(s.ok and s.value > 0 for s in stats) < 1:
            F.skip(name, f"no finite {col} for any model in {_rel(rd.path)}")
            continue
        stats = [s if (s.ok and s.value > 0) else Stat() for s in stats]       # log axis: zero is "not measured", never plotted
        fig = grouped_bars([MODEL_LABELS[m] for m in models], [(ylabel, stats)], ylabel, f"Computational cost: {ylabel.split(' (')[0].lower()}",
                           rd.prov, log=True, ylim=None)
        fig.axes[0].get_legend().remove()
        srcs = [rd.file("ablation_timings.csv" if df is tim else "ablation_runs.csv")]
        F.add(name, fig, srcs, col, rd.prov, "mean over folds of fold values, via the framework's aggregate_metric on the saved raw rows", UNC,
              rd.hash)


def _scal(rd: ResultDir) -> Optional[pd.DataFrame]:
    df = read_csv(rd.file("scalability_summary.csv"))
    if df is None or not {"size", "component", "median_s"} <= set(df.columns):
        return None
    return df.assign(size=pd.to_numeric(df["size"], errors="coerce"), median_s=pd.to_numeric(df["median_s"], errors="coerce")).dropna(subset=["size"])


def _iqr(row: pd.Series) -> Optional[Tuple[float, float]]:
    lo, hi, m = _num(row.get("q25_s")), _num(row.get("q75_s")), _num(row.get("median_s"))
    return (max(m - lo, 0.0), max(hi - m, 0.0)) if math.isfinite(lo) and math.isfinite(hi) and math.isfinite(m) else None


def _run_note(rd: ResultDir) -> str:
    cfg = rd.manifest.get("config") or {}
    return f"single outer fold {cfg.get('fold', '?')}"


def fig_scalability(rd: ResultDir, F: Figures, min_sizes: int = 2) -> None:
    name = f"fig10_scalability__{rd.tag}"
    df = _scal(rd)
    if df is None:
        return F.skip(name, f"scalability_summary.csv missing or malformed in {_rel(rd.path)}")
    sizes = sorted(df["size"].unique())
    if len(sizes) < min_sizes:
        return F.skip(name, f"only {len(sizes)} measured size(s) ({', '.join(str(int(s)) for s in sizes)}) in {_rel(rd.path)}; "
                            f"a runtime-vs-size curve needs at least {min_sizes}")
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    drawn = 0
    for i, (comp, lab) in enumerate(SCAL_COMPONENTS):
        sub = df[(df.component == comp) & df.median_s.notna() & (df.median_s > 0)].drop_duplicates("size").set_index("size")
        if sub.empty:
            continue
        # one slot per measured size of the experiment; an unmeasured (component, size) is NaN, so the line is BROKEN there
        # rather than joined across the gap (no interpolation)
        ys = np.array([sub.loc[z, "median_s"] if z in sub.index else NAN for z in sizes], dtype=float)
        errs = np.array([(_iqr(sub.loc[z]) or (0.0, 0.0)) if z in sub.index else (0.0, 0.0) for z in sizes]).T
        ax.errorbar(np.array(sizes, dtype=float), ys, yerr=errs, marker="o", capsize=2.5, color=COLORS[i % len(COLORS)], label=lab, lw=1.4)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return F.skip(name, f"none of the plotted components were measured in {_rel(rd.path)}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(int(s)) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Graph size (nodes; measured sizes only)")
    ax.set_ylabel("Runtime (s), median (bars: q25–q75 over repeats)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _title(ax, f"Scalability ({_run_note(rd)})", short_prov(rd.prov))
    F.add(name, fig, [rd.file("scalability_summary.csv")], "median_s per component", rd.prov,
          "median over repeats as saved in scalability_summary.csv", "interquartile range (q25_s–q75_s) over repeats as saved", rd.hash)


def fig_component_cost(rd: ResultDir, F: Figures) -> None:
    name = f"fig09d_component_cost__{rd.tag}"
    df = _scal(rd)
    if df is None:
        return F.skip(name, f"scalability_summary.csv missing in {_rel(rd.path)}")
    comps = [("chi_construction", "χ construction"), ("megits_adjacency", "MeGiTS adjacency"), ("centrality_ranking", "Ranking (eigenvector centrality)"),
             ("path_tracing", "Path tracing"), ("attackg_fit", "AttacKG fit"), ("attackg_inference", "AttacKG inference"),
             ("ladder_fit", "LADDER fit"), ("ladder_inference", "LADDER inference")]
    size = df["size"].max()
    sub = df[df["size"] == size].set_index("component")
    rows = [(lab, sub.loc[c]) for c, lab in comps if c in sub.index and _num(sub.loc[c, "median_s"]) > 0]
    if not rows:
        return F.skip(name, f"no ranking/baseline/path component measured in {_rel(rd.path)}")
    fig, ax = plt.subplots(figsize=(6.8, 0.4 * len(rows) + 2))
    y = np.arange(len(rows))
    errs = np.array([_iqr(r) or (0.0, 0.0) for _, r in rows]).T
    ax.errorbar([r["median_s"] for _, r in rows], y, xerr=errs, fmt="o", color=COLORS[0], capsize=3)   # points: log axis has no zero baseline
    ax.set_yticks(y)
    ax.set_yticklabels([l for l, _ in rows])
    ax.set_xscale("log")
    ax.set_xlabel("Runtime (s), median (bars: q25–q75)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    _title(ax, f"Component cost at {int(size)} nodes ({_run_note(rd)})", short_prov(rd.prov))
    F.add(name, fig, [rd.file("scalability_summary.csv")], "median_s", rd.prov, "median over repeats as saved",
          "interquartile range (q25_s–q75_s) as saved", rd.hash)


# ------------------------------------------------------------------------------------------------ driver
def build_all(results_dir: Path, out: Path, dpi: int = 300, formats: Sequence[str] = ("pdf", "png"), min_sizes: int = 2,
              include_dry_runs: bool = False, allow_unknown_provenance: bool = False) -> Figures:
    F = Figures(out, dpi, formats)
    dirs = discover(results_dir)
    n_found = len(dirs)
    F.remove_stale()
    if not include_dry_runs:
        for d in [d for d in dirs if is_dry_run(d, results_dir)]:
            F.skip(f"all figures from {_rel(d.path)}", "dry-run output (plumbing check, not an experiment); use --include-dry-runs to plot it")
        dirs = [d for d in dirs if not is_dry_run(d, results_dir)]
    if not allow_unknown_provenance:
        for d in [d for d in dirs if not d.prov_known]:
            F.skip(f"all figures from {_rel(d.path)}", "dataset provenance (synthetic / semi_synthetic / real) is not recorded in its manifest; "
                                                     "refusing to plot it unlabelled (--allow-unknown-provenance overrides)")
        dirs = [d for d in dirs if d.prov_known]
    sens = [d for d in dirs if re.match(r"^(alpha|tau)_", d.path.name)]
    main = [d for d in dirs if d not in sens]
    cmp_dirs = [d for d in main if d.kind in ("model_comparison", "external_baseline_comparison")]
    abl_dirs = [d for d in main if d.kind == "ablation_study"]
    for d in cmp_dirs:
        fig_model_comparison(d, F)
        fig_per_type(d, F, "macro_f1", "Macro-F1")
        fig_per_type(d, F, "micro_f1", "Micro-F1")
        fig_paths(d, F)
        fig_topk(d, F)
        fig_cost(d, F)
    for d in abl_dirs:
        fig_ablation(d, F)
        fig_chi(d, F, "macro_f1", "Macro-F1")
        fig_chi(d, F, "micro_f1", "Micro-F1")
        fig_label_fraction(d, F)
        fig_depth(d, F)
        fig_per_type(d, F, "macro_f1", "Macro-F1")
    for d in main:
        if d.kind == "scalability":
            fig_scalability(d, F, min_sizes)
            fig_component_cost(d, F)
    fig_sensitivity(sens, F)
    for p in sorted(Path(results_dir).rglob("ranked_iocs.csv")):
        if "figures" in p.parts or (not include_dry_runs and _dry_path(p, results_dir)):
            continue
        if not allow_unknown_provenance:
            F.skip(f"fig08b from {_rel(p)}", "a ranking-run output carries no recorded dataset provenance (--allow-unknown-provenance overrides)")
            continue
        fig_ranking_distribution(p, F)
    if not n_found:
        F.skip("all", f"no experiment output (manifest.json) found under {_rel(results_dir)}")
    required = {"fig01 model comparison": cmp_dirs, "fig02-05 ablation / χ / label fraction / depth": abl_dirs,
                "fig10 scalability": [d for d in main if d.kind == "scalability"]}
    for k, v in required.items():
        if not v:
            F.skip(k, "no saved results of this experiment type under " + _rel(results_dir))
    F.write_manifests()
    return F


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None, help="default: <results-dir>/figures")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--min-scalability-sizes", type=int, default=2,
                    help="a runtime-vs-size figure needs this many measured sizes (default 2)")
    ap.add_argument("--include-dry-runs", action="store_true")
    ap.add_argument("--allow-unknown-provenance", action="store_true",
                    help="plot results whose manifest does not record the dataset provenance (labelled as such)")
    a = ap.parse_args(argv)
    rdir = Path(a.results_dir)
    F = build_all(rdir, Path(a.out) if a.out else rdir / "figures", a.dpi, a.formats, a.min_scalability_sizes, a.include_dry_runs, a.allow_unknown_provenance)
    F.close()
    print(f"{len({r['figure_filename'].rsplit('.', 1)[0] for r in F.rows})} figure(s), {len(F.skipped)} skipped -> {F.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
