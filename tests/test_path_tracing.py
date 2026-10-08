"""Attack-path tracing and path-level evaluation: toy graphs with known answers, directionality, schema validity, cycle
prevention, partial overlap, Exact-Path Hit@5, deterministic ranking, reference-path leakage, and the dev datasets.
Synthetic / semi-synthetic data only - nothing here is a manuscript result."""
import inspect
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iocevaluator.datasets import load_dataset
from iocevaluator.evaluation import EvalConfig, evaluate_dataset, write_evaluation
from iocevaluator.labels import ReferencePath
from iocevaluator.path_tracing import (PathConfig, StructureSupport, edge_f1_typed, evaluate_case, exact_match,
                                       relation_f1, resolve_reference, summarise_cases, trace_case, trace_paths)
from iocevaluator.supervised_gcn import SupervisedGCNConfig
from iocevaluator.synthetic_tikg import generate, write_dataset
from iocevaluator.tikg import TIKG, TIKGError, Entity, EntityType as ET, TA, D, F

FIXTURE = Path(__file__).parent / "fixtures" / "cve_pool_sample.json"

ENTS = [("ta1", ET.THREAT_ACTOR), ("v1", ET.VULNERABILITY), ("v2", ET.VULNERABILITY), ("d1", ET.DEVICE),
        ("p1", ET.PLATFORM), ("f1", ET.FILE), ("m1", ET.ATTACK_METHOD)]
TRIPS = [("ta1", "exploits", "v1"), ("d1", "affected_by", "v1"), ("p1", "affected_by", "v1"), ("f1", "contains", "v1"),
         ("d1", "runs_on", "p1"), ("ta1", "unauthorised_access", "d1"), ("v1", "evolves_to", "v2"),
         ("m1", "exploits", "v1"), ("ta1", "uses", "m1")]


def toy(trips=TRIPS, ents=ENTS, strict=True):
    return TIKG([Entity(i, t, campaign="k") for i, t in ents], trips, strict=strict)


def vec(g, **kw):
    a = np.zeros(g.n)
    for k, v in kw.items():
        a[g.index[k]] = v
    return a


def run(g, seeds, risk=None, cfg=PathConfig(max_edges=2), conf_other=0.3):
    conf = np.full(g.n, conf_other)
    for s in seeds:
        conf[g.index[s]] = 1.0
    risk = np.zeros(g.n) if risk is None else risk
    return trace_case(g, "k", conf, risk, None, None, cfg)


def ids(g, p):
    return [g.entities[i].id for i in p.nodes]


# ------------------------------------------------------------------------------------------- toy known answers
def test_toy_two_edge_paths_between_seeds_known_answer():
    g = toy()
    cfg = PathConfig(max_edges=2, weights=(("priority", 1), ("confidence", 0), ("structure_weight", 0), ("coherence", 0)))
    res = run(g, ["ta1", "p1"], vec(g, ta1=0.9, p1=0.5), cfg)
    got = {tuple(ids(g, p)): (p.relations, p.directions) for p in res.candidates}
    assert got == {("ta1", "v1", "p1"): (("exploits", "affected_by"), ("fwd", "rev")),
                   ("ta1", "d1", "p1"): (("unauthorised_access", "runs_on"), ("fwd", "fwd"))}
    assert [g.entities[i].id for i in res.seeds] == ["ta1", "p1"]               # ordered by priority R, then id
    # equal priority components -> deterministic tie-break on the node-id sequence ("d1" < "v1")
    assert [ids(g, p)[1] for p in res.candidates] == ["d1", "v1"]
    # the priority component decides when the middle node differs: v1 more prioritised -> v1 path first
    up = run(g, ["ta1", "p1"], vec(g, ta1=0.9, p1=0.5, v1=0.8), cfg)
    assert [ids(g, p)[1] for p in up.candidates] == ["v1", "d1"]
    assert up.candidates[0].components["priority"] == pytest.approx((0.9 + 0.8 + 0.5) / 3)
    assert up.candidates[0].score == pytest.approx((0.9 + 0.8 + 0.5) / 3)        # only the priority weight is non-zero


def test_score_is_the_weighted_mean_of_its_components():
    g = toy()
    res = run(g, ["ta1", "p1"], vec(g, ta1=0.9, p1=0.5), PathConfig(max_edges=2))
    for p in res.candidates:
        c = p.components
        assert set(c) == {"priority", "confidence", "structure_weight", "coherence"}
        assert all(0.0 <= v <= 1.0 for v in c.values())
        assert p.score == pytest.approx((c["priority"] + c["confidence"] + c["structure_weight"]) / 3)   # default lambdas
        assert c["confidence"] == pytest.approx((1.0 + 0.3 + 1.0) / 3)           # seeds 1.0, the middle node 0.3
    with pytest.raises(ValueError):
        PathConfig(weights=(("priority", 0), ("confidence", 0), ("structure_weight", 0), ("coherence", 0)))
    with pytest.raises(ValueError):
        PathConfig(weights=(("priority", 1),))


def test_start_node_rule_confidence_threshold_and_priority_top_n():
    g = toy()
    conf = np.array([0.9, 0.2, 0.2, 0.6, 0.4, 0.5, 0.1])                     # ta1, v1, v2, d1, p1, f1, m1
    risk = vec(g, ta1=0.1, d1=0.7, f1=0.4)
    from iocevaluator.path_tracing import select_seeds
    scope = np.arange(g.n)
    s = select_seeds(g, scope, conf, risk, PathConfig(conf_threshold=0.5))
    assert [g.entities[i].id for i in s] == ["d1", "f1", "ta1"]                # conf >= 0.5, ordered by R desc
    top = select_seeds(g, scope, conf, risk, PathConfig(conf_threshold=0.5, priority_top_n=2))
    assert [g.entities[i].id for i in top] == ["d1", "f1"]
    assert len(select_seeds(g, scope, conf, risk, PathConfig(conf_threshold=0.95))) == 0
    assert run(g, [], None).candidates == []                                   # no high-confidence node -> no paths


# ------------------------------------------------------------------------------------------- directionality
def test_direction_and_relation_are_preserved_in_both_readings():
    g = toy()
    ta_first = run(g, ["ta1", "p1"], vec(g, ta1=0.9, p1=0.1)).candidates
    p_first = run(g, ["ta1", "p1"], vec(g, ta1=0.1, p1=0.9)).candidates         # priority decides the reading order
    a = next(p for p in ta_first if ids(g, p)[1] == "d1")
    b = next(p for p in p_first if ids(g, p)[1] == "d1")
    assert (ids(g, a), a.relations, a.directions) == (["ta1", "d1", "p1"], ("unauthorised_access", "runs_on"), ("fwd", "fwd"))
    assert (ids(g, b), b.relations, b.directions) == (["p1", "d1", "ta1"], ("runs_on", "unauthorised_access"), ("rev", "rev"))
    assert a.typed_edges(g) == list(reversed(b.typed_edges(g)))                 # same stored triplets either way
    assert a.describe(g) == "ta1 -unauthorised_access-> d1 -runs_on-> p1"
    assert b.describe(g) == "p1 <-runs_on- d1 <-unauthorised_access- ta1"
    audit = a.audit(g)
    assert [(r["from"], r["relation"], r["to"], r["direction"], r["triplet"]) for r in audit] == \
        [("ta1", "unauthorised_access", "d1", "fwd", "T8"), ("d1", "runs_on", "p1", "fwd", "T7")]
    # a reference that is the same path read backwards is NOT an exact match in strict mode, but is in "either" mode
    ref = resolve_reference(g, ReferencePath("c", ["p1", "d1", "ta1"], ["runs_on", "unauthorised_access"], "k"))
    assert ref.valid and ref.directions == ("rev", "rev")
    assert not exact_match(g, a, ref) and exact_match(g, a, ref, either=True) and exact_match(g, b, ref)
    # a wrong relation name for an existing pair makes the reference invalid (no silent relation rewriting)
    bad = resolve_reference(g, ReferencePath("c", ["ta1", "v1", "p1"], ["exploits", "runs_on"], "k"))
    assert not bad.valid and not exact_match(g, a, bad)


# ------------------------------------------------------------------------------------------- schema validity
def test_invalid_schema_edges_are_rejected_and_non_chi_steps_never_traced():
    with pytest.raises(TIKGError):
        toy(TRIPS + [("ta1", "exploits", "d1")])                               # TA -exploits-> D is not a Table-2 triplet
    assert len(toy(TRIPS + [("ta1", "exploits", "d1")], strict=False).triplets) == len(TRIPS)   # dropped when non-strict
    # X1 = <TA, uses, F> is a TIKG extension but is not a step of any chi_1..chi_20 template -> never traced
    g = TIKG([Entity(i, t, campaign="k") for i, t in ENTS], TRIPS + [("ta1", "uses", "f1")])
    sup = StructureSupport()
    assert not sup.supports_step((TA, "uses", F, "fwd")) and not sup.supports_step((F, "uses", TA, "rev"))
    assert sup.supports_step((TA, "unauthorised_access", D, "fwd")) and sup.supports_step((D, "unauthorised_access", TA, "rev"))
    res = run(g, ["ta1", "f1"], vec(g, ta1=0.9, f1=0.5), PathConfig(max_edges=4))
    assert res.candidates                                                       # reachable through T1 / T4 instead
    for p in res.candidates:
        for h in p.audit(g):
            assert {h["from_type"], h["to_type"]} != {"threat_actor", "file"}   # the X1 hop is never traversed


def test_template_support_is_stricter_than_step_support():
    g = toy()
    cfg_step = PathConfig(max_edges=2)
    cfg_tpl = PathConfig(max_edges=2, support="template")
    seeds = ["ta1", "p1", "v1"]
    risk = vec(g, ta1=0.9, v1=0.5, p1=0.1)
    step = {tuple(ids(g, p)) for p in run(g, seeds, risk, cfg_step).candidates}
    tpl = {tuple(ids(g, p)) for p in run(g, seeds, risk, cfg_tpl).candidates}
    assert tpl < step                                                         # strictly fewer
    assert ("ta1", "d1", "p1") in tpl                                         # TA->D->P is a segment of chi_17
    assert ("ta1", "v1", "p1") in step and ("ta1", "v1", "p1") not in tpl     # TA->V<-P belongs to no single template
    with pytest.raises(ValueError):
        PathConfig(support="other")


# ------------------------------------------------------------------------------------------- cycles / explosion / determinism
def test_paths_are_simple_and_bounded():
    trips = TRIPS + [("p1", "affected_by", "v2"), ("d1", "affected_by", "v2")]      # extra cycles
    g = toy(trips)
    seeds = [e for e, _ in ENTS]
    for max_edges in (3, 5, 8):
        res = run(g, seeds, vec(g, **{s: 0.1 * i for i, s in enumerate(seeds)}), PathConfig(max_edges=max_edges))
        assert res.candidates
        for p in res.candidates:
            assert len(set(p.nodes)) == len(p.nodes)                              # no repeated node => no cycle
            assert 2 <= p.n_edges <= max_edges and len(p.relations) == len(p.directions) == len(p.nodes) - 1
        keys = [p.key(g) for p in res.candidates]
        assert len(keys) == len(set(keys))                                        # each path reported once
        seen = {tuple(ids(g, p)) for p in res.candidates}                         # each undirected path appears once,
        assert not any(tuple(ids(g, p))[::-1] in seen for p in res.candidates)    # read from its higher-priority endpoint
    small = run(g, seeds, None, PathConfig(max_edges=8, max_candidates=5, max_paths_per_seed=3))
    assert len(small.candidates) <= 5 and small.truncated
    tiny = run(g, seeds, None, PathConfig(max_edges=8, max_expansions_per_seed=4))
    assert tiny.truncated


def test_ranking_is_deterministic_and_independent_of_input_order():
    seeds = ["ta1", "p1", "v1", "d1"]
    risk_map = dict(ta1=0.9, p1=0.4, v1=0.4, d1=0.4)
    outs = []
    for k in range(4):
        trips, ents = list(TRIPS), list(ENTS)
        random.Random(k).shuffle(trips)
        random.Random(100 + k).shuffle(ents)
        g = toy(trips, ents)
        res = run(g, seeds, vec(g, **risk_map), PathConfig(max_edges=4))
        outs.append([(p.describe(g), round(p.score, 10)) for p in res.candidates])
    assert outs[0] == outs[1] == outs[2] == outs[3] and len(outs[0]) > 6
    g = toy()
    assert [p.describe(g) for p in run(g, seeds, vec(g, **risk_map), PathConfig(max_edges=4)).candidates] == [d for d, _ in outs[0]]


# ------------------------------------------------------------------------------------------- metrics
def test_edge_f1_partial_overlap_hand_computed():
    pred = [("a", "r1", "b"), ("b", "r2", "c")]
    ref = [("a", "r1", "b"), ("c", "r3", "d")]
    assert edge_f1_typed(pred, ref) == pytest.approx(0.5)                       # tp 1, P 1/2, R 1/2
    assert edge_f1_typed(pred, pred) == 1.0 and edge_f1_typed(pred, [("x", "r", "y")]) == 0.0 and edge_f1_typed([], ref) == 0.0
    assert edge_f1_typed(pred, [("a", "r9", "b"), ("b", "r2", "c")]) == pytest.approx(0.5)   # relation type must match
    assert edge_f1_typed(pred[:1], ref) == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert relation_f1(["x", "y"], ["x", "z"]) == pytest.approx(0.5)
    assert relation_f1(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(2 / 3)    # multiset overlap


def test_evaluate_case_exact_hit_at_5_edge_f1_and_rank():
    g = toy()
    cfg = PathConfig(max_edges=4, top_k=5, weights=(("priority", 1), ("confidence", 0), ("structure_weight", 0), ("coherence", 0)))
    res = run(g, ["ta1", "p1"], vec(g, ta1=0.9, p1=0.5), cfg)
    assert len(res.candidates) >= 6
    # reference = the candidate at rank r (strict reading): hit iff r <= 5, ref_rank reports r
    for rank in (1, 3, 5, 6):
        c = res.candidates[rank - 1]
        ref = ReferencePath("c", ids(g, c), list(c.relations), "k")
        row = evaluate_case(g, ref, res, cfg)
        assert row["ref_rank"] == rank and row["exact_hit"] == (1.0 if rank <= 5 else 0.0)
        assert row["exact_hit_strict"] == row["exact_hit"] and row["ref_valid"]
        assert row["edge_f1"] == pytest.approx(edge_f1_typed(res.candidates[0].typed_edges(g), c.typed_edges(g)))
    # Edge-F1 of the TOP candidate against a 3-edge reference: hand-computed 2 * (1/2) * (1/3) / (5/6) = 0.4
    top = res.candidates[0]
    assert ids(g, top) == ["ta1", "d1", "p1"]
    ref = ReferencePath("c", ["ta1", "d1", "v1", "p1"], ["unauthorised_access", "affected_by", "affected_by"], "k")
    row = evaluate_case(g, ref, res, cfg)
    assert row["edge_f1"] == pytest.approx(0.4)                                # tp 1 of 2 predicted, 1 of 3 reference
    assert row["relation_f1"] == pytest.approx(2 * (1 / 2) * (1 / 3) / (1 / 2 + 1 / 3))  # ua matches; runs_on vs affected_by x2
    # Hit@k respects k
    ref5 = ReferencePath("c", ids(g, res.candidates[5]), list(res.candidates[5].relations), "k")
    assert evaluate_case(g, ref5, res, PathConfig(max_edges=4, top_k=6, weights=cfg.weights))["exact_hit"] == 1.0
    assert evaluate_case(g, ref5, res, PathConfig(max_edges=4, top_k=5, weights=cfg.weights))["exact_hit"] == 0.0
    # reversed reading: strict miss, "either" hit; per-case flags
    c = res.candidates[0]
    rev = ReferencePath("c", ids(g, c)[::-1], list(c.relations)[::-1], "k")
    row = evaluate_case(g, rev, res, cfg)
    assert row["exact_hit_strict"] == 0.0 and row["exact_hit_either"] == 1.0 and row["exact_hit"] == 0.0
    assert evaluate_case(g, rev, res, PathConfig(max_edges=4, match="either", weights=cfg.weights))["exact_hit"] == 1.0
    # no candidates -> miss, Edge-F1 0
    empty = run(g, [], None, cfg)
    row = evaluate_case(g, ref, empty, cfg)
    assert row["exact_hit"] == 0.0 and row["edge_f1"] == 0.0 and row["n_candidates"] == 0
    # a reference longer than the tracing limit is flagged and cannot be hit
    long = ReferencePath("c", ["ta1", "d1", "v1", "v2"], ["unauthorised_access", "affected_by", "evolves_to"], "k")
    assert evaluate_case(g, long, res, PathConfig(max_edges=2))["ref_within_max_edges"] is False


def test_summarise_cases_buckets_and_deterministic_intervals():
    rows = [{"case_id": f"c{i}", "seed": s, "bucket": b, "exact_hit": h, "exact_hit_either": h, "edge_f1": e, "relation_f1": e}
            for i, (b, h, e) in enumerate([("2", 1.0, 1.0), ("2", 0.0, 0.5), ("3", 1.0, 0.8), (">=4", 0.0, 0.2)]) for s in (0, 1)]
    s = summarise_cases(pd.DataFrame(rows))
    assert s["overall"]["n_cases"] == 4 and s["2"]["n_cases"] == 2 and s["3"]["n_cases"] == 1 and s[">=4"]["n_cases"] == 1
    assert s["2"]["exact_hit"]["mean"] == pytest.approx(0.5) and s["overall"]["exact_hit"]["mean"] == pytest.approx(0.5)
    assert s["overall"]["edge_f1"]["mean"] == pytest.approx((1 + 0.5 + 0.8 + 0.2) / 4)
    assert s["overall"]["exact_hit"]["std_over_seeds"] == pytest.approx(0.0)
    assert summarise_cases(pd.DataFrame(rows)) == s


# ------------------------------------------------------------------------------------------- no reference-path leakage
def test_tracing_has_no_access_to_reference_paths():
    for fn in (trace_paths, trace_case):
        params = set(inspect.signature(fn).parameters)
        assert not any("ref" in p for p in params), fn
    g = toy()
    a = run(g, ["ta1", "p1"], vec(g, ta1=0.9, p1=0.5), PathConfig(max_edges=4))
    # whatever references exist elsewhere, tracing the same case gives byte-identical candidates
    b = run(g, ["ta1", "p1"], vec(g, ta1=0.9, p1=0.5), PathConfig(max_edges=4))
    assert [(p.key(g), p.score) for p in a.candidates] == [(p.key(g), p.score) for p in b.candidates]


# ------------------------------------------------------------------------------------------- dev datasets (harness integration)
@pytest.fixture(scope="module", params=["synthetic", "semi_synthetic"])
def ds(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    d = generate("dev", 42) if request.param == "synthetic" else \
        generate("dev", 42, vulnerability_source="real", cve_pool=FIXTURE)
    write_dataset(d, root / "dev")
    return load_dataset(request.param, profile="dev", root=root)


PCFG = PathConfig(max_edges=4, conf_threshold=0.0, priority_top_n=8)
ECFG = EvalConfig(seeds=(0,), model=SupervisedGCNConfig(max_epochs=8, lr=1e-2), paths=PCFG)


def test_reference_paths_resolve_against_the_tikg_and_file_directions_agree(ds):
    assert len(ds.reference_paths) == 76
    for r in ds.reference_paths:
        rr = resolve_reference(ds.tikg, r)
        assert rr.valid and list(rr.relations) == r.relations                     # every reference edge exists, same relation
        assert r.directions is not None and list(rr.directions) == r.directions  # directions derived from the graph == file


@pytest.fixture(scope="module")
def result(ds):
    return evaluate_dataset(ds, ECFG, use_reference_paths=True)


def test_path_level_evaluation_runs_on_dev_dataset(ds, result):
    paths, runs = result.paths, result.runs
    assert len(result.folds) == 10 and len(paths) == 76                           # every case once per seed (1 seed)
    assert paths.case_id.nunique() == 76 and paths.groupby("case_id").fold.nunique().max() == 1
    camp = ds.tikg.campaigns()
    for _, r in paths.iterrows():                                                  # each case evaluated in its HELD-OUT fold
        f = result.folds[int(r.fold)]
        assert r.campaign in set(camp[f.test_nodes_mask]) and r.campaign not in set(camp[f.train])
    for c in ("exact_hit", "exact_hit_strict", "exact_hit_either", "edge_f1", "relation_f1"):
        assert paths[c].between(0, 1).all()
    assert (paths.exact_hit_either >= paths.exact_hit_strict).all() and paths.ref_valid.all()
    assert paths.n_candidates.gt(0).any() and paths.top1_path.str.len().gt(0).any()
    assert {"exact_path_hit5", "path_edge_f1", "exact_path_hit5_2edges", "exact_path_hit5_3edges",
            "exact_path_hit5_ge4edges", "n_path_cases"} <= set(runs.columns)
    assert runs.n_path_cases.sum() == 76
    s = result.summary()["paths"]
    assert s["overall"]["n_cases"] == 76 and (s["2"]["n_cases"], s["3"]["n_cases"], s[">=4"]["n_cases"]) == (32, 26, 18)
    assert 0 <= s["overall"]["exact_hit"]["mean"] <= 1
    assert result.summary()["overall"]["exact_path_hit5"]["n_folds"] <= 10


def test_candidates_are_auditable_schema_valid_and_simple(ds, result):
    g = ds.tikg
    sup = StructureSupport()
    from iocevaluator.tikg import TRIPLET_SIGNATURES
    sigs = set(TRIPLET_SIGNATURES.values())
    row = result.paths[result.paths.top1_path.str.len() > 0].iloc[0]
    assert "->" in row.top1_path or "<-" in row.top1_path
    ref = ds.reference_paths[0]
    # re-trace one case and check every candidate against the schema
    from iocevaluator.megits import megits_adjacency
    adj = megits_adjacency(g).adj
    conf, risk = np.ones(g.n), np.random.RandomState(0).rand(g.n)
    tr = trace_case(g, ref.campaign, conf, risk, adj, None, PCFG, sup)
    assert tr.candidates
    for p in tr.candidates:
        assert len(set(p.nodes)) == len(p.nodes) and PCFG.min_edges <= p.n_edges <= PCFG.max_edges
        for step in p.audit(g):
            assert step["triplet"].startswith("T")                                 # a Table-2 triplet in stored direction
            a, b = g.entities[g.index[step["from"]]].type, g.entities[g.index[step["to"]]].type
            tri = (a, step["relation"], b) if step["direction"] == "fwd" else (b, step["relation"], a)
            assert tri in sigs
        assert all(g.campaigns()[i] == ref.campaign for i in p.nodes)              # confined to the case scope


def test_reference_paths_never_influence_training_or_path_generation(ds, result):
    """Same data without / with scrambled reference paths: classification metrics, candidate lists and GCN outputs are
    identical - the references only enter the scoring of the candidate lists."""
    plain = evaluate_dataset(ds, ECFG, use_reference_paths=False)
    common = [c for c in plain.runs.columns if c in result.runs.columns and c != "ranker_alpha"]   # alpha is a ranker diagnostic
    assert "macro_f1" in common and len(common) == len(plain.runs.columns) - 1
    pd.testing.assert_frame_equal(plain.runs[common], result.runs[common], check_exact=True)
    scr = []
    rng = random.Random(0)
    for r in ds.reference_paths:
        nodes = list(r.nodes)
        rng.shuffle(nodes)
        scr.append(ReferencePath(r.case_id, nodes, [], r.campaign))               # garbage references, same campaigns
    from iocevaluator.evaluation import evaluate_cv
    other = evaluate_cv(ds.tikg, ds.labels, ECFG, severity=ds.severity, severity_known=ds.severity_known, reference_paths=scr,
                        dataset_info=result.manifest["dataset"])
    cols = ["fold", "seed", "case_id", "n_seeds", "n_candidates", "top_paths", "top_scores", "top1_path", "truncated"]
    pd.testing.assert_frame_equal(result.paths[cols].reset_index(drop=True), other.paths[cols].reset_index(drop=True))
    assert result.manifest["paths"]["reference_paths"]["role"] == "evaluation only"


def test_path_evaluation_is_deterministic_and_written(ds, result, tmp_path):
    again = evaluate_dataset(ds, ECFG, use_reference_paths=True)
    pd.testing.assert_frame_equal(result.paths, again.paths)
    pd.testing.assert_frame_equal(result.runs, again.runs)
    assert result.manifest["content_sha256"] == again.manifest["content_sha256"]
    out = write_evaluation(tmp_path, result)
    assert out["results_path_cases.csv"].exists() and len(pd.read_csv(out["results_path_cases.csv"])) == 76
    m = result.manifest["paths"]
    assert m["enabled"] and m["max_edges"] == 4 and m["support"] == "step" and m["weights"]["coherence"] == 0
    assert m["reference_paths"]["n_cases"] == 76 and m["reference_paths"]["sha256"]
