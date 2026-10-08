"""Semi-synthetic mode (real public CVE metadata + synthetic graph).  No test touches the network."""
import copy
import json
import re
import shutil
import socket
from collections import Counter
from pathlib import Path

import pytest

from iocevaluator.datasets import SyntheticDataError, load_dataset
from iocevaluator.synthetic_tikg import check_reproducible, generate, validate, write_dataset
from iocevaluator.synthetic_tikg.cve_pool import load_pool, normalise_nvd_item, select_cves
from iocevaluator.synthetic_tikg.generator import dumps

FIXTURE = Path(__file__).parent / "fixtures" / "cve_pool_sample.json"
REAL_POOL = Path("data/raw/cve/cve_pool.json")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*a, **k):
        raise AssertionError("tests must not use the network")
    monkeypatch.setattr(socket.socket, "connect", deny)


@pytest.fixture(scope="module")
def semi():
    return generate("dev", 42, vulnerability_source="real", cve_pool=FIXTURE)


def _fails(rep):
    return {c.name for c in rep.failures()}


def test_semi_dev_validates_and_counts(semi):
    rep = validate(semi)
    assert rep.ok, rep.summary()
    assert (len(semi.nodes), len(semi.edges), len(semi.attack_paths)) == (350, 1091, 76)
    assert Counter(n["type"] for n in semi.nodes)["vulnerability"] == 210


def test_metadata_declares_semi_synthetic(semi):
    m = semi.metadata
    assert m["dataset_type"] == "semi_synthetic" and m["real_component"] == "public_CVE_metadata"
    assert m["purpose"] == "development_and_pipeline_validation" and m["not_for_reported_experimental_results"] is True
    assert set(m["synthetic_components"]) == {"campaigns", "threat_actors", "graph_relationships", "labels",
                                              "reference_attack_paths"}
    assert m["cve_snapshot"]["sha256"] == load_pool(FIXTURE)["sha256"] and m["seed"] == 42


def test_real_cves_and_provenance(semi):
    pool = {r["cve_id"]: r for r in load_pool(FIXTURE)["records"]}
    for n in semi.nodes:
        if n["type"] == "vulnerability":
            assert n["provenance"] == "real_public" and re.fullmatch(r"CVE-\d{4}-\d{4,}", n["name"])
            r = pool[n["name"]]                                     # metadata is copied, never invented
            assert n["attrs"]["cvss_base_score"] == r["cvss_base_score"] and n["attrs"]["cwe"] == r["cwe"]
            assert n["attrs"]["text"] == r["description"] and "target_countries" not in n["attrs"]
        else:
            assert n["provenance"] == "synthetic" and n["attrs"]["synthetic"] is True
    assert all(e["provenance"] == "synthetic" for e in semi.edges)
    assert all(re.fullmatch(r"actor:SYN-TA-\d{4}", n["id"]) for n in semi.nodes if n["type"] == "threat_actor")


def test_structure_preserved(semi):
    st = validate(semi).stats
    assert st["edge_locality"]["campaign"] > 0.5 and st["edge_locality"]["cross_family"] < 0.2
    sep = st["similarity_separation"]
    assert sep["jaccard_related_campaigns"] > 2 * sep["jaccard_unrelated_campaigns"]


def test_reproducible_byte_identical(tmp_path, semi):
    assert check_reproducible("dev", 42, vulnerability_source="real", cve_pool=FIXTURE)
    write_dataset(generate("dev", 42, vulnerability_source="real", cve_pool=FIXTURE), tmp_path / "a")
    write_dataset(semi, tmp_path / "b")
    for f in ("nodes", "edges", "labels", "campaigns", "attack_paths", "metadata"):
        assert (tmp_path / "a" / f"{f}.json").read_bytes() == (tmp_path / "b" / f"{f}.json").read_bytes()
    other = generate("dev", 7, vulnerability_source="real", cve_pool=FIXTURE)
    assert dumps(other.edges) != dumps(semi.edges)


def test_modes_are_distinguishable_and_synthetic_mode_unchanged(semi):
    syn = generate("dev", 42)
    assert syn.metadata["dataset_type"] == "synthetic" and "provenance" not in syn.nodes[0]
    assert validate(syn).ok
    assert any(n["name"].startswith("CVE-209") for n in syn.nodes if n["type"] == "vulnerability")
    assert not any(n["name"].startswith("CVE-209") for n in semi.nodes if n["type"] == "vulnerability")


def test_loader_kinds(tmp_path, semi):
    write_dataset(semi, tmp_path / "semi" / "dev")
    write_dataset(generate("dev", 42), tmp_path / "syn" / "dev")
    ds = load_dataset("semi_synthetic", profile="dev", root=tmp_path / "semi")
    assert ds.is_synthetic and ds.is_semi_synthetic and ds.source == "semi_synthetic"
    assert ds.severity_known.sum() > 0 and "NOT FOR REPORTED RESULTS" in ds.banner()
    with pytest.raises(SyntheticDataError):
        ds.require_real()
    with pytest.raises(SyntheticDataError):                           # kinds cannot be mixed up
        load_dataset("synthetic", profile="dev", root=tmp_path / "semi")
    with pytest.raises(SyntheticDataError):
        load_dataset("semi_synthetic", profile="dev", root=tmp_path / "syn")


@pytest.mark.parametrize("mutate,check", [
    (lambda d: d.edges[0].pop("provenance"), "all_edges_provenance_synthetic"),
    (lambda d: d.nodes.__setitem__(0, {**d.nodes[0], "provenance": "real_public"}), "non_vulnerability_nodes_provenance_synthetic"),
    (lambda d: next(n for n in d.nodes if n["type"] == "vulnerability").update(name="CVE-xx"), "vulnerability_ids_are_real_cve_format"),
    (lambda d: next(n for n in d.nodes if n["type"] == "vulnerability")["attrs"].update(target_countries=["GB"]),
     "no_synthetic_facts_attached_to_real_cves"),
    (lambda d: next(n for n in d.nodes if n["type"] == "threat_actor").update(name="APT28"), "no_real_threat_actor_attribution"),
    (lambda d: d.metadata.pop("cve_snapshot"), "cve_snapshot_recorded"),
    (lambda d: [n for n in d.nodes if n["type"] == "vulnerability"][0]["attrs"].update(text=""), "cve_metadata_fields_present_or_null"),
    (lambda d: d.nodes.pop(0), "node_counts_exact_by_type"),
])
def test_semi_validation_detects_corruption(semi, mutate, check):
    d = copy.deepcopy(semi)
    mutate(d)
    assert check in _fails(validate(d))


# ----------------------------------------------------------------------------------------------- pool helpers
def test_normalise_nvd_item_keeps_nulls():
    item = {"cve": {"id": "CVE-2020-0001", "sourceIdentifier": "x@y", "published": "2020-01-01T00:00:00.000",
                    "lastModified": "2020-02-01T00:00:00.000", "vulnStatus": "Analyzed",
                    "descriptions": [{"lang": "en", "value": "Heap overflow in Foo PLC firmware."}],
                    "references": [{"url": "https://example.org/a"}]}}
    r = normalise_nvd_item(item)
    assert r["cvss_base_score"] is None and r["cvss_vector"] is None and r["severity"] is None
    assert r["cwe"] == [] and r["cpe"] == [] and r["references"] == ["https://example.org/a"]
    assert normalise_nvd_item({"cve": {**item["cve"], "descriptions": [{"lang": "en", "value": "** REJECT ** x"}]}}) is None
    assert normalise_nvd_item({"cve": {**item["cve"], "id": "not-a-cve"}}) is None


def test_selection_deterministic_vendor_capped_and_errors_when_short():
    pool = load_pool(FIXTURE)
    a, b = select_cves(pool, 150), select_cves(pool, 150)
    assert [r["cve_id"] for r in a] == [r["cve_id"] for r in b] and len(a) == 150
    cap = int(0.06 * 150)
    assert max(Counter(r["vendors"][0] for r in a if r["vendors"]).values()) <= cap
    with pytest.raises(ValueError):
        select_cves(pool, pool["n_records"] + 1)


@pytest.mark.skipif(not REAL_POOL.exists(), reason="full CVE cache not present")
def test_manuscript_scale_with_full_cache():
    d = generate("manuscript_scale", 42, vulnerability_source="real")
    rep = validate(d)
    assert rep.ok, rep.summary()
    assert (len(d.nodes), len(d.edges), len(d.attack_paths)) == (3728, 11624, 76)
    assert len({n["name"] for n in d.nodes if n["type"] == "vulnerability"}) == 2298
