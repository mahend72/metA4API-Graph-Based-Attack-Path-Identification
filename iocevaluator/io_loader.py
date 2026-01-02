from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .schemas import Indicator, ThreatRecord

log = logging.getLogger(__name__)


def _parse_dt(s: str) -> datetime:
    """Parse common ISO-8601 strings found in OTX-like feeds."""
    # OTX often uses: 2023-01-01T12:34:56.123
    # Some feeds omit fractional seconds.
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # last resort: fromisoformat
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception as e:
        raise ValueError(f"Unrecognized datetime format: {s}") from e


def load_threats_json(path: str | Path) -> List[ThreatRecord]:
    """Load normalized ThreatRecord objects from a JSON file.

    Expected (flexible) schema per record:
      - name (str)
      - created (str ISO)
      - revision (int, optional)
      - adversary (str, optional)
      - industries (list[str], optional)
      - targeted_countries (list[str], optional)
      - attack_ids (list[str], optional)
      - malware_families (list[str], optional)
      - indicators (list[dict], optional) where each dict has:
          indicator, type, title, description, content

    This matches what the original notebook expects from `ICS_IOCs_Updated.json`.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a list at root of {path}, got {type(raw)}")

    out: List[ThreatRecord] = []
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        indicators: List[Indicator] = []
        for ind in rec.get("indicators", []) or []:
            if not isinstance(ind, dict):
                continue
            indicators.append(
                Indicator(
                    indicator=str(ind.get("indicator", "")),
                    type=str(ind.get("type", "")),
                    title=str(ind.get("title", "")),
                    description=str(ind.get("description", "")),
                    content=str(ind.get("content", "")),
                )
            )

        created_raw = rec.get("created") or rec.get("created_at") or rec.get("createdAt")
        if not created_raw:
            # If absent, keep a stable default rather than crashing.
            created = datetime(1970, 1, 1)
        else:
            created = _parse_dt(str(created_raw))

        out.append(
            ThreatRecord(
                name=str(rec.get("name", rec.get("title", "unknown"))),
                created=created,
                revision=int(rec.get("revision", rec.get("modified", 0)) or 0),
                adversary=str(rec.get("adversary", rec.get("author_name", "")) or ""),
                industries=list(rec.get("industries", []) or []),
                targeted_countries=list(rec.get("targeted_countries", rec.get("targetedCountries", [])) or []),
                attack_ids=list(rec.get("attack_ids", rec.get("attackIds", [])) or []),
                malware_families=list(rec.get("malware_families", rec.get("malwareFamilies", [])) or []),
                indicators=indicators,
                raw=rec,
            )
        )

    log.info("Loaded %d threat records from %s", len(out), path)
    return out


def save_threats_json(path: str | Path, threats: List[ThreatRecord]) -> None:
    """Save ThreatRecord objects as JSON (for reproducibility)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(t: ThreatRecord) -> Dict[str, Any]:
        return {
            "name": t.name,
            "created": t.created.isoformat(),
            "revision": t.revision,
            "adversary": t.adversary,
            "industries": t.industries,
            "targeted_countries": t.targeted_countries,
            "attack_ids": t.attack_ids,
            "malware_families": t.malware_families,
            "indicators": [
                {
                    "indicator": i.indicator,
                    "type": i.type,
                    "title": i.title,
                    "description": i.description,
                    "content": i.content,
                }
                for i in t.indicators
            ],
        }

    with path.open("w", encoding="utf-8") as f:
        json.dump([to_dict(t) for t in threats], f, indent=2)
