import json

import numpy as np
import pandas as pd

from iocevaluator.experiments import (VARIANTS, ExperimentConfig, ABLATION_VARIANTS, evaluate_external_oof, run_cv,
                                      write_outputs)
from iocevaluator.labels import build_label_space
from iocevaluator.models import TrainConfig
from iocevaluator.splits import group_stratified_folds
from iocevaluator.tikg_features import FeatureConfig
from .synthetic import make_synthetic


def small_cfg(**kw):
    return ExperimentConfig(n_splits=3, seeds=(0, 1), train=TrainConfig(max_epochs=120, patience=20, lr=3e-2, hidden=16),
                            features=FeatureConfig(), **kw)


def test_variant_registry_covers_manuscript_experiments():
    assert {"metA4API", "gat", "hgt", "original_adjacency_gcn", "binary_semantic_gcn", "megits_paths_only",
            "megits_graphs_only", "metA4API_1layer", "metA4API_3layer", "metA4API_4layer"} <= set(VARIANTS)
    assert all(f"chi{k}" in VARIANTS for k in range(1, 21))
    assert VARIANTS["metA4API_4layer"].layers == 4


def test_run_cv_end_to_end_and_outputs(tmp_path):
    g, lab = make_synthetic(9)
    ls = build_label_space(g, lab)
    cfg = small_cfg()
    names = ["metA4API", "original_adjacency_gcn", "gat", "hgt", "chi19", "metA4API_3layer"]
    res = run_cv(g, ls, names, cfg, verbose=False)
    df = res.frame()
    assert len(df) == 3 * 2 * len(names)                          # folds x seeds x variants
    assert {"macro_f1", "micro_f1", "roc_auc", "pr_auc", "train_time_per_epoch_s", "inference_latency_ms"} <= set(df.columns)
    assert df[df.variant == "metA4API"].micro_f1.mean() > 0.5       # learns the synthetic structure
    assert df[df.variant == "metA4API"].centrality_time_s.notna().all() and df[df.variant == "gat"].centrality_time_s.isna().all()
    assert np.isfinite(res.oof["metA4API"][np.concatenate([f.test for f in res.folds])]).all()
    s = res.summary()
    assert "wilcoxon_vs_metA4API" in s["gat"] and s["metA4API"]["macro_f1"]["std"] >= 0
    man = write_outputs(tmp_path, res, g, ls, cfg, names)
    for f in ["results_per_run.csv", "summary.json", "folds.json", "manifest.json", "oof_probs_metA4API.npy"]:
        assert (tmp_path / f).exists()
    assert man["backend"].startswith("PyTorch") and len(man["structures"]) == 20
    assert [s["id"] for s in man["structures"]] == list(range(1, 21))
    assert {s["provenance"] for s in man["structures"]} == {"manuscript_fig3"} and all(s["formula"] for s in man["structures"])
    # external-method scoring path (AttacKG / LADDER outputs would be supplied this way)
    ext = evaluate_external_oof("external", res.oof["metA4API"], ls, res.folds)
    assert len(ext) == 3


def test_reduced_label_fraction_uses_fewer_training_nodes():
    g, lab = make_synthetic(9)
    ls = build_label_space(g, lab)
    full = run_cv(g, ls, ["metA4API"], small_cfg(max_folds=1), verbose=False).frame()
    quarter = run_cv(g, ls, ["metA4API"], small_cfg(max_folds=1, label_fraction=0.25), verbose=False).frame()
    assert quarter.n_train.iloc[0] < 0.3 * full.n_train.iloc[0] + 1


def test_cv_runs_are_reproducible():
    g, lab = make_synthetic(9)
    ls = build_label_space(g, lab)
    a = run_cv(g, ls, ["metA4API"], small_cfg(max_folds=1), verbose=False).frame().macro_f1.values
    b = run_cv(g, ls, ["metA4API"], small_cfg(max_folds=1), verbose=False).frame().macro_f1.values
    np.testing.assert_allclose(a, b)
