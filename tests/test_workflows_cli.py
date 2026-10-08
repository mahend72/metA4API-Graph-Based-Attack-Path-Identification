import json
from pathlib import Path

import numpy as np

from iocevaluator.cli import main
from iocevaluator.workflows import rank
from .synthetic import make_synthetic

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path):
    g, lab = make_synthetic(9)
    g.save(tmp_path / "tikg.json")
    (tmp_path / "labels.json").write_text(json.dumps({"labels": lab}))
    return g


def test_cli_build_tikg_from_sample(tmp_path):
    main(["build-tikg", "--input", str(ROOT / "data" / "sample_threats.json"), "--out", str(tmp_path / "t.json")])
    assert json.loads((tmp_path / "t.json").read_text())["entities"]


def test_cli_evaluate_then_rank_with_paths(tmp_path):
    g = _write(tmp_path)
    main(["evaluate", "--tikg", str(tmp_path / "tikg.json"), "--labels", str(tmp_path / "labels.json"),
          "--out", str(tmp_path / "ev"), "--variants", "main", "--folds", "3", "--seeds", "1", "--max-folds", "1"])
    assert (tmp_path / "ev" / "manifest.json").exists()
    (tmp_path / "sev.json").write_text(json.dumps({"cve0_0": 9.8}))
    (tmp_path / "ref.json").write_text(json.dumps([{"case_id": "c", "campaign": "c0",
        "nodes": ["ta0_0", "dev0_0", "cve0_0", "dev0_1", "ta0_1"], "relations": []}]))
    s = rank(tmp_path / "tikg.json", tmp_path / "labels.json", tmp_path / "rk", severity_path=tmp_path / "sev.json",
             reference_paths=tmp_path / "ref.json", train_cfg=__import__("iocevaluator.models", fromlist=["x"]).TrainConfig(max_epochs=20))
    assert s["n_unknown_severity"] == g.n - 1 and "in-sample" in s["predictions"]
    for f in ["ranked_iocs.csv", "ranked_paths.json", "explanations.json", "rank_summary.json"]:
        assert (tmp_path / "rk" / f).exists()
    assert "overall" in s["path_evaluation"]
