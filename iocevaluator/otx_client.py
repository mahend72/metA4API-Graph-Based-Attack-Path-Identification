from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from OTXv2 import OTXv2

from .schemas import Indicator, ThreatRecord

log = logging.getLogger(__name__)


def fetch_otx_pulses(api_key: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch pulses from AlienVault OTX.

    This uses `OTXv2.getall()` (as in the original notebook). Depending on your
    account/rate limits, you may wish to set `limit`.
    """
    otx = OTXv2(api_key)
    pulses = otx.getall()
    if limit is not None:
        pulses = pulses[:limit]
    log.info("Fetched %d pulses from OTX", len(pulses))
    return pulses


def normalize_pulse(p: Dict[str, Any]) -> ThreatRecord:
    """Convert a raw OTX pulse into our ThreatRecord schema.

    Notes:
    - OTX fields vary between endpoints; this mapper is intentionally defensive.
    - Indicators are best-effort; unknown types are preserved.
    """
    created_raw = p.get("created") or p.get("created_at") or p.get("createdAt") or ""
    created: datetime
    try:
        created = datetime.fromisoformat(str(created_raw).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        created = datetime(1970, 1, 1)

    indicators: List[Indicator] = []
    for ind in p.get("indicators", []) or []:
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

    return ThreatRecord(
        name=str(p.get("name", p.get("title", "unknown"))),
        created=created,
        revision=int(p.get("revision", 0) or 0),
        adversary=str(p.get("adversary", p.get("author_name", "")) or ""),
        industries=list(p.get("industries", []) or []),
        targeted_countries=list(p.get("targeted_countries", p.get("targetedCountries", [])) or []),
        attack_ids=list(p.get("attack_ids", p.get("attackIds", [])) or []),
        malware_families=list(p.get("malware_families", p.get("malwareFamilies", [])) or []),
        indicators=indicators,
        raw=p,
    )


def fetch_and_normalize_from_env(api_key_env: str = "OTX_API_KEY", limit: Optional[int] = None) -> List[ThreatRecord]:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing OTX API key. Set env var {api_key_env} (see .env.example).\n"
            "Tip: never commit API keys to GitHub."
        )
    pulses = fetch_otx_pulses(api_key, limit=limit)
    return [normalize_pulse(p) for p in pulses]
