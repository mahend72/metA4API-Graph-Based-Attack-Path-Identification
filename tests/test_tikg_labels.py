import json
from pathlib import Path

import numpy as np
import pytest

from iocevaluator.io_loader import load_threats_json
from iocevaluator.labels import build_label_space, load_severity, load_reference_paths
from iocevaluator.tikg import (TIKG, Entity, EntityType as ET, TIKGError, TRIPLET_SIGNATURES, RELATIONS,
                               load_tikg, tikg_from_threats)
from .synthetic import make_synthetic

ROOT = Path(__file__).resolve().parents[1]


def test_schema_has_seven_types_nine_relations_eleven_triplets():
    assert len(ET) == 7 and len(RELATIONS) == 9
    assert [k for k in TRIPLET_SIGNATURES if k.startswith("T")] == [f"T{i}" for i in range(1, 12)]


def test_invalid_triplet_rejected_and_unknown_entity_rejected():
    ents = [Entity("a", ET.THREAT_ACTOR), Entity("v", ET.VULNERABILITY)]
    TIKG(ents, [("a", "exploits", "v")])
    with pytest.raises(TIKGError):
        TIKG(ents, [("v", "exploits", "a")])          # wrong direction/signature
    with pytest.raises(TIKGError):
        TIKG(ents, [("a", "enables", "v")])           # `enables` has no signature in Table 2
    with pytest.raises(TIKGError):
        TIKG(ents, [("a", "exploits", "zzz")])
    with pytest.raises(TIKGError):
        TIKG(ents + [Entity("a", ET.DEVICE)], [])


def test_Q_shapes_directions_and_exclude():
    g, _ = make_synthetic(3)
    q = g.Q(ET.DEVICE, "affected_by", ET.VULNERABILITY)
    assert q.shape == (len(g.type_nodes[ET.DEVICE]), len(g.type_nodes[ET.VULNERABILITY]))
    assert q.nnz == 6
    ex = np.zeros(g.n, bool)
    ex[g.index["dev0_0"]] = True
    assert g.Q(ET.DEVICE, "affected_by", ET.VULNERABILITY, exclude=ex).nnz == 5
    f = g.Q(ET.THREAT_ACTOR, "uses", ET.FILE, tgt_subtypes=["domain"])
    assert f.nnz == 6                                  # 3 campaigns x 2 actors


def test_roundtrip_and_stats(tmp_path):
    g, _ = make_synthetic(3)
    p = tmp_path / "g.json"
    g.save(p)
    g2 = load_tikg(p)
    assert g2.n == g.n and len(g2.triplets) == len(g.triplets)
    s = g.stats()
    assert s["n_nodes"] == g.n and s["avg_undirected_degree"] == pytest.approx(2 * len(g.triplets) / g.n)


def test_otx_adapter_only_creates_explicit_relations():
    threats = load_threats_json(ROOT / "data" / "sample_threats.json")
    g = tikg_from_threats(threats)
    kinds = {t.sig for t in g.triplets}
    assert kinds <= {"T1", "T11", "X1"}               # nothing about devices/platforms is invented
    assert len(g.type_nodes[ET.DEVICE]) == 0 and len(g.type_nodes[ET.PLATFORM]) == 0
    assert any(e.type == ET.THREAT_ACTOR for e in g.entities)


def test_label_space_valid_masks_and_unlabelled():
    g, lab = make_synthetic(3)
    drop = next(iter(lab))
    partial = {k: v for k, v in lab.items() if k != drop}
    ls = build_label_space(g, partial)
    assert not ls.labelled[g.index[drop]] and ls.labelled.sum() == g.n - 1
    # label columns exist per type; a node is only scored on its own type's columns
    i = g.index["ta0_0"]
    cols = ls.type_columns(ET.THREAT_ACTOR)
    assert ls.valid[i].sum() == len(cols) and ls.Y[i, cols].sum() == 1
    with pytest.raises(ValueError):
        build_label_space(g, {"nope": ["x"]})


def test_severity_and_reference_loaders(tmp_path):
    g, _ = make_synthetic(2)
    (tmp_path / "s.json").write_text(json.dumps({"cve0_0": 9.8, "unknown": 5}))
    sev, known = load_severity(g, tmp_path / "s.json")
    assert sev[g.index["cve0_0"]] == pytest.approx(0.98) and known.sum() == 1       # no imputation
    (tmp_path / "r.json").write_text(json.dumps([{"case_id": "x", "nodes": ["a", "b"], "relations": ["uses"]}]))
    assert load_reference_paths(tmp_path / "r.json")[0].nodes == ["a", "b"]
