"""Plot pipeline: reads saved results only, honest about NaN / missing values, provenance-labelled, manifest-linked.

Fixtures are written with the framework's own ``summarise`` / ``paired_comparison`` so the file schemas are the real ones."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iocevaluator.ablation import REFERENCE, paired_comparison
from iocevaluator.evaluation import summarise

_spec = importlib.util.spec_from_file_location("plot_results", Path(__file__).resolve().parents[1] / "scripts" / "plot_results.py")
pr = importlib.util.module_from_spec(_spec)
sys.modules["plot_results"] = pr
_spec.loader.exec_module(pr)

FOLDS, SEEDS = 6, 2
ETYPES = ["threat_actor", "vulnerability", "attack_method", "file", "attack_type", "device", "platform"]


def _runs(variants, base=0.3, nan_cols=None):
    rng = np.random.RandomState(1)
    rows, trows = [], []
    for vi, v in enumerate(variants):
        for f in range(FOLDS):
            for s in range(SEEDS):
                r = {"variant": v, "fold": f, "seed": s, "n_parameters": 1000 + 10 * vi}
                for m in ("macro_f1", "micro_f1", "roc_auc", "pr_auc", "top5_hit_rate", "top10_hit_rate", "exact_path_hit5", "path_edge_f1",
                          "path_relation_f1", "exact_path_hit5_2edges", "path_edge_f1_2edges"):
                    r[m] = base + 0.02 * vi + 0.01 * f + 0.001 * s + 0.003 * rng.rand()
                for m in (nan_cols or {}).get(v, []):
                    r[m] = float("nan")
                rows.append(r)
                for t in ETYPES:
                    trows.append({"variant": v, "fold": f, "seed": s, "entity_type": t, "macro_f1": 0.2 + 0.05 * vi + 0.01 * f,
                                  "micro_f1": 0.25 + 0.01 * f, "roc_auc": 0.6, "pr_auc": 0.4})
    return pd.DataFrame(rows), pd.DataFrame(trows)


def _write(d: Path, kind, variants, source="semi_synthetic", nan_cols=None, paired=False, extra_manifest=None):
    d.mkdir(parents=True, exist_ok=True)
    runs, bt = _runs(variants, nan_cols=nan_cols)
    summ = {v: summarise(runs[runs.variant == v].drop(columns="variant"), bt[bt.variant == v].drop(columns="variant")) for v in variants}
    (d / "ablation_summary.json").write_text(json.dumps({"provenance": {"source": source}, "reference_variant": variants[0] if kind != "ablation_study" else REFERENCE,
                                                        "variants": summ}, sort_keys=True), encoding="utf-8")
    man = {"kind": kind, "provenance": {"source": source, "protocol_frozen": False}, "content_sha256": "abc123" + d.name,
           "variants": [{"name": v} for v in variants], **(extra_manifest or {})}
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    runs.to_csv(d / "ablation_runs.csv", index=False)
    tim = runs[["variant", "fold", "seed"]].assign(train_time_total_s=1.5, inference_time_s=0.2)
    tim.to_csv(d / "ablation_timings.csv", index=False)
    if paired:
        paired_comparison(runs, REFERENCE).to_csv(d / "ablation_paired.csv", index=False)
    return summ


def _build(tmp_path, **kw):
    out = tmp_path / "figs"
    F = pr.build_all(tmp_path / "res", out, dpi=60, **kw)
    return F, out


def _bar_heights(fig):
    ax = fig.axes[0]
    return [[p.get_height() for p in c.patches] for c in ax.containers if hasattr(c, "patches")]


def _ablation_dir(tmp_path, **kw):
    names = [v for v, _ in pr.ABLATION_ORDER] + [f"chi{k}" for k in range(1, 21)] + ["megits_full_1layer", "megits_full_3layer", "megits_full_lf25",
                                                                                          "megits_full_lf50", "megits_full_lf75"]
    return _write(tmp_path / "res" / "abl", "ablation_study", names, **kw)


def test_figures_read_saved_values_not_constants(tmp_path):
    summ = _write(tmp_path / "res" / "cmp", "external_baseline_comparison", ["gcn", "gat", "hgt", "attackg", "ladder"])
    F, _ = _build(tmp_path)
    fig = F.figs["fig01_model_comparison__cmp"]
    for i, h in enumerate(_bar_heights(fig)):               # series = models, groups = the 4 metrics
        m = ["gcn", "gat", "hgt", "attackg", "ladder"][i]
        assert h == pytest.approx([summ[m]["overall"][k]["fold_mean"] for k, _ in pr.PRIMARY])
    F.close()
    # change the saved file -> the figure changes
    p = tmp_path / "res" / "cmp" / "ablation_summary.json"
    d = json.loads(p.read_text())
    d["variants"]["gcn"]["overall"]["macro_f1"]["fold_mean"] = 0.987654
    p.write_text(json.dumps(d))
    F2, _ = _build(tmp_path)
    assert _bar_heights(F2.figs["fig01_model_comparison__cmp"])[0][0] == pytest.approx(0.987654)
    F2.close()


def test_uncertainty_is_the_saved_ci95_folds_t(tmp_path):
    summ = _write(tmp_path / "res" / "cmp", "model_comparison", ["gcn", "gat", "hgt"])
    F, _ = _build(tmp_path)
    ax = F.figs["fig01_model_comparison__cmp"].axes[0]
    cont = [c for c in ax.containers if hasattr(c, "patches")][0]
    segs = cont.errorbar.lines[2][0].get_segments()
    agg = summ["gcn"]["overall"]["macro_f1"]
    assert segs[0][0][1] == pytest.approx(agg["ci95_folds_t"][0]) and segs[0][1][1] == pytest.approx(agg["ci95_folds_t"][1])
    F.close()
    # the manifest names the same interval
    row = pd.read_csv(tmp_path / "figs" / "figure_manifest.csv").iloc[0]
    assert "ci95_folds_t" in row.uncertainty_method
    # the plotted interval follows the file, not a recomputation
    p = tmp_path / "res" / "cmp" / "ablation_summary.json"
    d = json.loads(p.read_text())
    m = d["variants"]["gcn"]["overall"]["macro_f1"]
    m["ci95_folds_t"] = [m["fold_mean"] - 0.2, m["fold_mean"] + 0.3]
    p.write_text(json.dumps(d))
    F2, _ = _build(tmp_path)
    s2 = [c for c in F2.figs["fig01_model_comparison__cmp"].axes[0].containers if hasattr(c, "patches")][0].errorbar.lines[2][0].get_segments()[0]
    assert s2[1][1] - s2[0][1] == pytest.approx(0.5)
    F2.close()


def test_nan_metric_is_not_plotted_as_zero(tmp_path):
    _write(tmp_path / "res" / "cmp", "external_baseline_comparison", ["gcn", "gat", "hgt", "attackg", "ladder"],
           nan_cols={"attackg": ["roc_auc", "pr_auc"]})
    p = tmp_path / "res" / "cmp" / "ablation_summary.json"
    d = json.loads(p.read_text())
    for k in ("roc_auc", "pr_auc"):                         # what json.dumps writes for a NaN aggregate
        d["variants"]["attackg"]["overall"][k] = {"fold_mean": float("nan"), "ci95_folds_t": [float("nan")] * 2, "n_folds": 0}
    p.write_text(json.dumps(d))
    F, _ = _build(tmp_path)
    fig = F.figs["fig01_model_comparison__cmp"]
    h = _bar_heights(fig)[3]                                # AttacKG
    assert math.isnan(h[2]) and math.isnan(h[3]) and not math.isnan(h[0])
    assert [t.get_text() for t in fig.axes[0].texts].count("n/a") == 2
    F.close()


def test_missing_metrics_and_files_do_not_crash(tmp_path):
    d = tmp_path / "res" / "cmp"
    _write(d, "external_baseline_comparison", ["gcn", "gat"])
    s = json.loads((d / "ablation_summary.json").read_text())
    s["variants"]["gcn"]["overall"] = {}                     # no metrics at all for one model
    s["variants"]["gat"].pop("by_entity_type")
    (d / "ablation_summary.json").write_text(json.dumps(s))
    for f in ("ablation_timings.csv", "ablation_runs.csv"):
        (d / f).unlink()
    (tmp_path / "res" / "empty").mkdir()
    (tmp_path / "res" / "empty" / "manifest.json").write_text("{not json")
    F, out = _build(tmp_path)
    assert (out / "figure_manifest.csv").exists() and (out / "figures_skipped.csv").exists()
    assert len(F.skipped) > 0
    F.close()
    # nothing at all
    F2 = pr.build_all(tmp_path / "nothing", tmp_path / "o2", dpi=50)
    assert F2.rows == [] and any("no experiment output" in s["reason"] for s in F2.skipped)


def test_semi_synthetic_is_labelled_everywhere(tmp_path):
    _write(tmp_path / "res" / "cmp", "external_baseline_comparison", ["gcn", "gat", "hgt", "attackg", "ladder"], source="semi_synthetic")
    _ablation_dir(tmp_path, source="semi_synthetic", paired=True)
    F, out = _build(tmp_path)
    man = pd.read_csv(out / "figure_manifest.csv")
    assert len(man) and man.dataset_provenance.str.contains("Semi-synthetic benchmark").all()
    for name, fig in F.figs.items():
        texts = " ".join(t.get_text() for a in fig.axes for t in [a.title, *a.texts]) + " ".join(t.get_text() for t in fig.texts)
        assert "Semi-synthetic benchmark" in texts, name
    F.close()


def test_synthetic_and_real_are_distinguished(tmp_path):
    _write(tmp_path / "res" / "a", "model_comparison", ["gcn", "gat"], source="synthetic")
    _write(tmp_path / "res" / "b", "model_comparison", ["gcn", "gat"], source="real")
    F, out = _build(tmp_path)
    man = pd.read_csv(out / "figure_manifest.csv")
    provs = dict(zip(man.figure_filename.str.extract(r"__(\w+)\.")[0], man.dataset_provenance))
    assert provs["a"].startswith("Synthetic") and provs["b"].startswith("Real data") and "protocol not frozen" in provs["b"]
    F.close()


def test_manifest_links_every_figure_to_existing_sources(tmp_path):
    _write(tmp_path / "res" / "cmp", "external_baseline_comparison", ["gcn", "gat", "hgt", "attackg", "ladder"])
    _ablation_dir(tmp_path, paired=True)
    F, out = _build(tmp_path)
    man = pd.read_csv(out / "figure_manifest.csv").fillna("")
    assert list(man.columns) == pr.MANIFEST_COLUMNS
    files = {p.name for p in out.iterdir() if p.suffix in (".pdf", ".png")}
    assert files == set(man.figure_filename) and len(files) > 0
    assert any(f.endswith(".pdf") for f in files) and any(f.endswith(".png") for f in files)
    for r in man.itertuples():
        srcs = [s for s in r.source_result_files.split(";") if s]
        assert srcs, r.figure_filename
        assert all(Path(s).exists() or (Path.cwd() / s).exists() for s in srcs)
        assert r.metrics and r.dataset_provenance and r.aggregation_method and r.uncertainty_method and r.config_hash
        assert r.config_hash.startswith("abc123")
    F.close()


def test_chi_significance_marks_come_only_from_saved_table(tmp_path):
    _ablation_dir(tmp_path, paired=True)
    pth = tmp_path / "res" / "abl" / "ablation_paired.csv"
    df = pd.read_csv(pth)
    df["significant_holm_0.05"] = False
    df["wilcoxon_p_holm"] = 0.5
    sel = (df.variant == "chi3") & (df.metric == "macro_f1")
    df.loc[sel, ["significant_holm_0.05", "wilcoxon_p_holm"]] = [True, 0.01]
    sel7 = (df.variant == "chi7") & (df.metric == "macro_f1")
    df.loc[sel7, ["significant_holm_0.05", "wilcoxon_p_holm"]] = [True, np.nan]      # untestable: must NOT be starred
    df.to_csv(pth, index=False)
    F, _ = _build(tmp_path)
    ax = F.figs["fig03_chi_contribution_macro_f1__abl"].axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    stars = [(round(t.xy[1]), labels[round(t.xy[1])]) for t in ax.texts if False] or \
            [labels[round(c.xy[1])] for c in ax.get_children() if c.__class__.__name__ == "Annotation" and c.get_text() == "*"]
    assert stars == ["χ3"]
    F.close()
    pth.unlink()                                             # no paired table -> no marks, no crash
    F2, _ = _build(tmp_path)
    ax2 = F2.figs["fig03_chi_contribution_macro_f1__abl"].axes[0]
    assert not [c for c in ax2.get_children() if c.__class__.__name__ == "Annotation" and c.get_text() == "*"]
    F2.close()


def test_ablation_family_figures_and_settings_from_manifest(tmp_path):
    _ablation_dir(tmp_path)
    F, _ = _build(tmp_path)
    for n in ("fig02_ablation__abl", "fig04_label_fraction__abl", "fig05_gcn_depth__abl", "fig06_per_entity_type_macro_f1__abl"):
        assert n in F.figs, n
    xs = sorted(F.figs["fig04_label_fraction__abl"].axes[0].lines[0].get_xdata())
    assert list(xs) == [25.0, 50.0, 75.0, 100.0]
    F.close()


def test_scalability_plots_only_measured_sizes_and_skips_single_size(tmp_path):
    d = tmp_path / "res" / "scal"
    d.mkdir(parents=True)
    rows = [{"size": s, "component": c, "median_s": 0.1 * s / 250 * (i + 1), "q25_s": 0.05, "q75_s": 0.2 * s / 250 * (i + 1)}
            for s in (250, 500, 1000) for i, (c, _) in enumerate(pr.SCAL_COMPONENTS)]
    rows = [r for r in rows if not (r["component"] == "path_tracing" and r["size"] == 500)]     # a size that was not measured
    pd.DataFrame(rows).to_csv(d / "scalability_summary.csv", index=False)
    (d / "manifest.json").write_text(json.dumps({"kind": "scalability", "provenance": {"source": "synthetic"}, "content_sha256": "h"}))
    F, _ = _build(tmp_path)
    ax = F.figs["fig10_scalability__scal"].axes[0]
    line = [c for c in ax.containers if c.get_label() == "Path tracing"][0][0]
    xs, ys = list(line.get_xdata()), list(line.get_ydata())
    assert xs == [250, 500, 1000] and math.isnan(ys[1]) and not math.isnan(ys[0])     # the 500 slot is a gap, not an interpolated value
    assert not [v for v in ys if v == 0]
    F.close()
    pd.DataFrame([r for r in rows if r["size"] == 250]).to_csv(d / "scalability_summary.csv", index=False)
    F2, _ = _build(tmp_path)
    assert "fig10_scalability__scal" not in F2.figs and any("only 1 measured size" in s["reason"] for s in F2.skipped)
    F2.close()


def test_dry_run_outputs_are_not_plotted_by_default(tmp_path):
    _write(tmp_path / "res" / "dryrun_x", "model_comparison", ["gcn", "gat"])
    F, _ = _build(tmp_path)
    assert F.rows == [] and any("dry-run" in s["reason"] for s in F.skipped)
    F.close()
    F2, _ = _build(tmp_path, include_dry_runs=True)
    assert F2.rows
    F2.close()


def test_script_does_not_embed_manuscript_numbers():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "plot_results.py").read_text(encoding="utf-8")
    import re
    assert not re.findall(r"\b0\.\d{3,}\b", src)             # no literal result-like constants


def test_unknown_provenance_is_refused_by_default(tmp_path):
    d = tmp_path / "res" / "cmp"
    _write(d, "model_comparison", ["gcn", "gat"])
    m = json.loads((d / "manifest.json").read_text())
    m.pop("provenance")
    (d / "manifest.json").write_text(json.dumps(m))
    s = json.loads((d / "ablation_summary.json").read_text())
    s.pop("provenance")
    (d / "ablation_summary.json").write_text(json.dumps(s))
    F, _ = _build(tmp_path)
    assert F.rows == [] and any("provenance" in x["reason"] for x in F.skipped)
    F.close()
    F2, _ = _build(tmp_path, allow_unknown_provenance=True)
    assert F2.rows and all("not recorded" in r["dataset_provenance"] for r in F2.rows)
    F2.close()


def test_dry_run_detected_below_results_root_and_by_run_class(tmp_path):
    _write(tmp_path / "res" / "dry_runs" / "cmp", "model_comparison", ["gcn", "gat"])
    _write(tmp_path / "res" / "cls", "model_comparison", ["gcn", "gat"], extra_manifest={"preflight": {"run_class": "dry_run"}})
    F, _ = _build(tmp_path)
    assert F.rows == []
    F.close()


def test_stale_figures_are_removed_so_folder_matches_manifest(tmp_path):
    _write(tmp_path / "res" / "cmp", "model_comparison", ["gcn", "gat"])
    F, out = _build(tmp_path)
    F.close()
    (out / "fig99_old__x.png").write_bytes(b"x")
    F2, out = _build(tmp_path)
    F2.close()
    man = pd.read_csv(out / "figure_manifest.csv")
    assert {p.name for p in out.glob("*.png")} | {p.name for p in out.glob("*.pdf")} == set(man.figure_filename)


def test_every_figure_type_carries_the_semi_synthetic_label(tmp_path):
    _write(tmp_path / "res" / "cmp", "external_baseline_comparison", ["gcn", "gat", "hgt", "attackg", "ladder"])
    _ablation_dir(tmp_path, paired=True)
    for n in ("alpha_0.3", "alpha_0.7", "tau_0.1", "tau_0.2"):
        _write(tmp_path / "res" / "sens" / n, "model_comparison", ["gcn"])
    F, out = _build(tmp_path)
    prefixes = {n.split("__")[0] for n in F.figs}
    for need in ("fig01_model_comparison", "fig02_ablation", "fig03_chi_contribution_macro_f1", "fig04_label_fraction", "fig05_gcn_depth",
                 "fig06_per_entity_type_macro_f1", "fig07_attack_path", "fig08a_prioritisation_topk", "fig08c_sensitivity_alpha",
                 "fig08c_sensitivity_tau", "fig09_train", "fig09_inference", "fig09_parameters"):
        assert need in prefixes, need
    for name, fig in F.figs.items():
        texts = " ".join(t.get_text() for a in fig.axes for t in [a.title, *a.texts]) + " ".join(t.get_text() for t in fig.texts)
        assert "Semi-synthetic benchmark" in texts, name
    assert "SENSITIVITY ANALYSIS" in F.figs["fig08c_sensitivity_alpha__sens"].axes[0].get_title(loc="left")
    F.close()
