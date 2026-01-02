from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class Indicator:
    """Atomic IOC / indicator as present in OTX pulses and many threat feeds."""
    indicator: str
    type: str
    title: str = ""
    description: str = ""
    content: str = ""


@dataclass
class ThreatRecord:
    """A single threat/pulse record (normalized)."""
    name: str
    created: datetime
    revision: int = 0
    adversary: str = ""
    industries: List[str] = field(default_factory=list)
    targeted_countries: List[str] = field(default_factory=list)
    attack_ids: List[str] = field(default_factory=list)
    malware_families: List[str] = field(default_factory=list)
    indicators: List[Indicator] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)  # original payload (optional)


@dataclass
class Event:
    """Co-occurrence bundle for building graphs.

    In this repo, each ThreatRecord typically becomes one Event.
    """
    attacker: str
    file_hash: List[str] = field(default_factory=list)
    email: List[str] = field(default_factory=list)
    ip: List[str] = field(default_factory=list)
    url: List[str] = field(default_factory=list)
    domain: List[str] = field(default_factory=list)
    hostname: List[str] = field(default_factory=list)
    region: List[str] = field(default_factory=list)  # targeted countries / regions
    attack_technique: List[str] = field(default_factory=list)
    vulnerability: List[str] = field(default_factory=list)  # CVEs etc

    def all_nodes(self) -> List[str]:
        nodes: List[str] = []
        nodes.append(self.attacker)
        nodes.extend(self.file_hash)
        nodes.extend(self.email)
        nodes.extend(self.ip)
        nodes.extend(self.url)
        nodes.extend(self.domain)
        nodes.extend(self.hostname)
        nodes.extend(self.region)
        nodes.extend(self.attack_technique)
        nodes.extend(self.vulnerability)
        return nodes
