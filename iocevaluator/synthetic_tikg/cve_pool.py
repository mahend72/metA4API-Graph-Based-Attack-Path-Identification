"""Real public CVE metadata (NVD API 2.0) -> normalised pool -> deterministic selection.

Only *vulnerability metadata* is real.  Nothing here says anything about threat actors.

Layout (``data/raw/cve/``):
    queries/<slug>.json   normalised records returned by one NVD query (resumable fetch cache)
    cve_pool.json         merged, de-duplicated normalised pool + snapshot description (sha256 recorded in metadata)

Missing metadata is stored as null / [] - nothing is imputed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
DEFAULT_CACHE = Path("data/raw/cve")
CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$")
POOL_FORMAT = 1

# (query, relevance tier, extra NVD params, max records).  Tier 3 = strongest ICS/AM relevance.
SEV = lambda s: {"cvssV3Severity": s}
QUERIES: List[Tuple[str, int, Dict[str, str], int]] = (
    [(q, 3, {}, 2000) for q in [
        "industrial control system", "SCADA", "programmable logic controller", "human machine interface",
        "Modbus", "OPC UA", "DNP3", "PROFINET", "EtherNet/IP", "3D printer", "additive manufacturing", "G-code",
        "STL file", "computer-aided design", "CAM software", "CNC", "industrial robot", "robot controller",
        "manufacturing execution system", "SIMATIC", "Modicon", "MELSEC", "ControlLogix", "FactoryTalk",
        "OctoPrint", "Marlin firmware", "Ultimaker", "Stratasys", "engineering workstation"]]
    + [(q, 2, d, 2000) for q in [
        "Siemens", "Rockwell Automation", "Schneider Electric", "Mitsubishi Electric", "Omron", "Beckhoff", "Moxa",
        "WAGO", "Phoenix Contact", "Advantech", "Delta Electronics", "ABB", "Honeywell", "Emerson", "Yokogawa",
        "Fanuc", "Autodesk", "SolidWorks", "PTC Creo", "Hitachi Energy", "Festo", "Pilz", "Siemens SIMATIC"]
       for d in [SEV("CRITICAL"), SEV("HIGH")]]
    + [(q, 1, d, 1000) for q in [
        "Windows remote desktop", "Windows SMB", "Linux kernel privilege escalation", "OpenSSL", "VxWorks",
        "industrial router", "industrial gateway", "historian", "firmware update", "embedded web server"]
       for d in [SEV("CRITICAL"), SEV("HIGH")]]
)

ICS_VENDORS = {"siemens", "rockwellautomation", "rockwell_automation", "schneider-electric", "schneiderelectric",
               "mitsubishielectric", "omron", "beckhoff", "moxa", "wago", "phoenixcontact", "advantech",
               "deltaww", "delta", "abb", "honeywell", "emerson", "yokogawa", "fanuc", "autodesk", "ptc",
               "dassault-systemes", "hitachi", "festo", "pilz", "stratasys", "ultimaker", "octoprint", "marlinfw",
               "3ds", "ge", "bentley", "eaton", "hms-networks", "ewon", "schweitzer_engineering_laboratories"}
ICS_TEXT = re.compile(r"\b(scada|plc|programmable logic|industrial control|ics\b|hmi|modbus|opc ua|profinet|"
                      r"3d print|additive manufactur|g-code|stl\b|cnc|cad\b|cam\b|robot|firmware|historian|"
                      r"manufactur|engineering workstation|fieldbus|dnp3|ethernet/ip)", re.I)


# ------------------------------------------------------------------------------------------------ normalisation
def _pick_cvss(metrics: dict) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[str]]:
    """Newest CVSS block available: v3.1 > v3.0 > v4.0 > v2.  Returns (version, score, vector, severity)."""
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV40", "cvssMetricV2"):
        blocks = metrics.get(key) or []
        if not blocks:
            continue
        blocks = sorted(blocks, key=lambda b: 0 if b.get("type") == "Primary" else 1)
        d = blocks[0].get("cvssData", {})
        sev = d.get("baseSeverity") or blocks[0].get("baseSeverity")
        return d.get("version"), d.get("baseScore"), d.get("vectorString"), (sev.upper() if sev else None)
    return None, None, None, None


def parse_cpe(cpe: str) -> Optional[dict]:
    p = cpe.split(":")
    if len(p) < 6 or p[0] != "cpe":
        return None
    return {"cpe": cpe, "part": p[2], "vendor": p[3], "product": p[4], "version": p[5] if p[5] not in ("*", "-") else None}


def normalise_nvd_item(item: dict) -> Optional[dict]:
    """One NVD 2.0 ``vulnerabilities[]`` element -> normalised record (None if rejected / no English text)."""
    c = item.get("cve", item)
    cid = c.get("id", "")
    if not CVE_RE.match(cid):
        return None
    desc = next((d["value"] for d in c.get("descriptions", []) if d.get("lang") == "en"), None)
    if not desc or desc.startswith("** REJECT") or desc.startswith("** RESERVED"):
        return None
    ver, score, vec, sev = _pick_cvss(c.get("metrics", {}))
    cwes: List[str] = []
    for w in sorted(c.get("weaknesses", []), key=lambda w: 0 if w.get("type") == "Primary" else 1):
        for d in w.get("description", []):
            if d.get("lang") == "en" and d["value"] not in cwes:
                cwes.append(d["value"])
    cpes, seen = [], set()
    for conf in c.get("configurations", []):
        for node in conf.get("nodes", []):
            for m in node.get("cpeMatch", []):
                if m.get("vulnerable") and m.get("criteria") not in seen:
                    seen.add(m["criteria"])
                    pc = parse_cpe(m["criteria"])
                    if pc:
                        cpes.append(pc)
    cpes = cpes[:20]
    return {
        "cve_id": cid, "description": desc, "published": c.get("published"), "last_modified": c.get("lastModified"),
        "vuln_status": c.get("vulnStatus"), "source_identifier": c.get("sourceIdentifier"),
        "cvss_version": ver, "cvss_base_score": score, "cvss_vector": vec, "severity": sev,
        "cwe": cwes, "cpe": cpes,
        "vendors": sorted({x["vendor"] for x in cpes}), "products": sorted({f"{x['vendor']}:{x['product']}" for x in cpes}),
        "platform_parts": sorted({x["part"] for x in cpes}),
        "references": [r["url"] for r in c.get("references", [])][:10],
    }


# ------------------------------------------------------------------------------------------------ fetching
def _slug(q: str, params: Dict[str, str]) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")
    return s + "".join(f"__{v.lower()}" for v in params.values())


def _get(url: str, retries: int = 6) -> dict:
    key = os.environ.get("NVD_API_KEY")
    req = urllib.request.Request(url, headers={"User-Agent": "metA4API-synthetic-tikg/1.0", **({"apiKey": key} if key else {})})
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            time.sleep(10 * (k + 1))
            last = e
    raise RuntimeError(f"NVD request failed: {url} ({last})")


def fetch_query(q: str, params: Dict[str, str], cap: int, delay: float) -> Tuple[List[dict], int, Optional[str]]:
    out, start, total, ts = [], 0, 0, None
    while start < min(cap, total or cap):
        qs = urllib.parse.urlencode({"keywordSearch": q, **params, "resultsPerPage": 2000, "startIndex": start})
        d = _get(f"{NVD_URL}?{qs}")
        total, ts = d.get("totalResults", 0), d.get("timestamp")
        out += [n for n in (normalise_nvd_item(v) for v in d.get("vulnerabilities", [])) if n]
        start += 2000
        time.sleep(delay)
    return out, total, ts


def fetch_all(cache: Path = DEFAULT_CACHE, refresh: bool = False, queries=QUERIES, log=print) -> None:
    """Resumable: a query whose cache file exists is skipped unless ``refresh``."""
    qdir = Path(cache) / "queries"
    qdir.mkdir(parents=True, exist_ok=True)
    delay = 0.8 if os.environ.get("NVD_API_KEY") else 6.5      # public NVD rate limit: 5 requests / 30 s
    for q, tier, params, cap in queries:
        f = qdir / f"{_slug(q, params)}.json"
        if f.exists() and not refresh:
            continue
        recs, total, ts = fetch_query(q, params, cap, delay)
        f.write_text(json.dumps({"query": q, "tier": tier, "params": params, "cap": cap, "total_results": total,
                                 "nvd_timestamp": ts, "records": recs}, ensure_ascii=False), encoding="utf-8")
        log(f"[nvd] {q!r} {params or ''}: {len(recs)} records (NVD total {total})")


def build_pool(cache: Path = DEFAULT_CACHE) -> Path:
    """Merge the per-query caches into ``cve_pool.json`` (sorted by id -> deterministic bytes)."""
    cache = Path(cache)
    merged: Dict[str, dict] = {}
    qinfo = []
    for f in sorted((cache / "queries").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        qinfo.append({"query": d["query"], "tier": d["tier"], "params": d["params"], "cap": d["cap"],
                      "total_results": d["total_results"], "returned": len(d["records"]), "nvd_timestamp": d["nvd_timestamp"]})
        for r in d["records"]:
            m = merged.setdefault(r["cve_id"], {**r, "matched_queries": [], "tier": 0})
            m["matched_queries"].append(d["query"])
            m["tier"] = max(m["tier"], d["tier"])
    recs = [merged[k] for k in sorted(merged)]
    for r in recs:
        r["matched_queries"] = sorted(set(r["matched_queries"]))
    ts = [q["nvd_timestamp"] for q in qinfo if q["nvd_timestamp"]]
    pool = {"format": POOL_FORMAT, "source": "NVD API 2.0 (https://nvd.nist.gov/developers/vulnerabilities)",
            "retrieved_between": [min(ts), max(ts)] if ts else None, "queries": qinfo, "n_records": len(recs),
            "records": recs}
    path = cache / "cve_pool.json"
    path.write_text(json.dumps(pool, ensure_ascii=False, indent=0) + "\n", encoding="utf-8", newline="\n")
    return path


# ------------------------------------------------------------------------------------------------ load / select
def load_pool(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "cve_pool.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run: python scripts/fetch_cve_pool.py")
    raw = p.read_bytes()
    pool = json.loads(raw.decode("utf-8"))
    pool["sha256"] = hashlib.sha256(raw).hexdigest()
    pool["path"] = str(p)
    return pool


def relevance(r: dict) -> float:
    """Documented ranking score (higher = more ICS / AM / OT relevant, better-described)."""
    s = 10.0 * r["tier"] + min(len(r["matched_queries"]), 5)
    vend = {v.lower() for v in r["vendors"]}
    if vend & ICS_VENDORS:
        s += 6
    if ICS_TEXT.search(r["description"]):
        s += 4
    if r["cvss_version"] and r["cvss_version"].startswith(("3", "4")):
        s += 3
    elif r["cvss_base_score"] is None:
        s -= 6
    if r["cwe"] and any(c.startswith("CWE-") for c in r["cwe"]):
        s += 1
    if r["cpe"]:
        s += 1
    return s


def select_cves(pool: dict, n: int, vendor_cap: float = 0.06) -> List[dict]:
    """Top-`n` by relevance, with at most ``vendor_cap * n`` CVEs per primary vendor (diversity guard); if the
    cap leaves too few, the highest-ranked skipped CVEs fill the remainder.  Pure function of (pool, n)."""
    ranked = sorted(pool["records"], key=lambda r: (-relevance(r), r["cve_id"]))
    cap = max(1, int(vendor_cap * n))
    per: Counter = Counter()
    chosen, skipped = [], []
    for r in ranked:
        v = r["vendors"][0] if r["vendors"] else None
        if len(chosen) < n and (v is None or per[v] < cap):
            chosen.append(r)
            per[v] += 1
        else:
            skipped.append(r)
    if len(chosen) < n:
        chosen += skipped[: n - len(chosen)]
    if len(chosen) < n:
        raise ValueError(f"CVE pool has only {len(chosen)} usable records, {n} required")
    return sorted(chosen, key=lambda r: r["cve_id"])
