from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .schemas import Event, ThreatRecord

log = logging.getLogger(__name__)


# A small, pragmatic mapping of indicator types to our buckets.
# OTX uses values like: IPv4, domain, URL, hostname, FileHash-MD5, FileHash-SHA256, email, CVE, etc.
TYPE_TO_BUCKET = {
    "ipv4": "ip",
    "ipv6": "ip",
    "ip": "ip",
    "domain": "domain",
    "hostname": "hostname",
    "url": "url",
    "email": "email",
    "cve": "vulnerability",
}


def _bucket_for_type(ind_type: str) -> str:
    t = (ind_type or "").strip().lower()
    if t.startswith("filehash"):
        return "file_hash"
    return TYPE_TO_BUCKET.get(t, "other")


def build_events(threats: Iterable[ThreatRecord]) -> List[Event]:
    """Convert threat records into graph events.

    We drop records without a usable attacker label, because the original
    notebook builds relations centered on `adversary` / attacker.
    """
    events: List[Event] = []
    dropped = 0

    for t in threats:
        attacker = (t.adversary or "").strip()
        if not attacker or attacker.lower() in {"unknown", "n/a"}:
            dropped += 1
            continue

        ev = Event(
            attacker=attacker,
            region=list({r for r in (t.targeted_countries or []) if r}),
            attack_technique=list({a for a in (t.attack_ids or []) if a}),
        )

        for ind in t.indicators:
            bucket = _bucket_for_type(ind.type)
            val = (ind.indicator or "").strip()
            if not val:
                continue
            if bucket == "file_hash":
                ev.file_hash.append(val)
            elif bucket == "email":
                ev.email.append(val)
            elif bucket == "ip":
                ev.ip.append(val)
            elif bucket == "url":
                ev.url.append(val)
            elif bucket == "domain":
                ev.domain.append(val)
            elif bucket == "hostname":
                ev.hostname.append(val)
            elif bucket == "vulnerability":
                ev.vulnerability.append(val)
            else:
                # keep unknowns out of the main graph by default
                pass

        events.append(ev)

    log.info("Built %d events (dropped %d records without attacker)", len(events), dropped)
    return events


def unique_attacker_list(events: Iterable[Event]) -> List[str]:
    return sorted({e.attacker for e in events})
