"""Generation profiles for the SYNTHETIC TIKG.

The node counts and #Class values of ``manuscript_scale`` are taken from manuscript Table 7 / Table 4.  The
per-relation edge quotas are NOT given in the manuscript (it only reports 11,624 relations in total); they are a
design choice of this generator and are documented as such in ``metadata.json``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

TYPES = ("threat_actor", "vulnerability", "attack_method", "file", "attack_type", "device", "platform")

# Manuscript Table 7 (#Class per entity type).
MANUSCRIPT_CLASS_COUNTS: Dict[str, int] = {
    "threat_actor": 11, "vulnerability": 47, "attack_method": 3, "file": 13,
    "attack_type": 11, "device": 14, "platform": 9,
}

# Manuscript Table 7 / Sec. 4.1 node counts (sum = 3,728).
MANUSCRIPT_NODE_COUNTS: Dict[str, int] = {
    "threat_actor": 271, "vulnerability": 2298, "attack_method": 68, "file": 336,
    "attack_type": 229, "device": 344, "platform": 182,
}

MANUSCRIPT_EDGES = 11624
MANUSCRIPT_NODES = 3728
MANUSCRIPT_EXPERT_NODES = 114

# Generator design choice (the manuscript gives no per-triplet breakdown).  Sums to 11,624.
MANUSCRIPT_EDGE_QUOTAS: Dict[str, int] = {
    "T1": 1500, "T2": 1900, "T3": 900, "T4": 700, "T5": 1100, "T6": 1500,
    "T7": 700, "T8": 900, "T9": 300, "T10": 900, "T11": 1224,
}
assert sum(MANUSCRIPT_EDGE_QUOTAS.values()) == MANUSCRIPT_EDGES
assert sum(MANUSCRIPT_NODE_COUNTS.values()) == MANUSCRIPT_NODES

# Manuscript Table path_length: 32 two-edge, 26 three-edge, 18 four-or-more-edge reference paths.
MANUSCRIPT_PATH_COUNTS: Tuple[int, int, int] = (32, 26, 18)


def scale_quotas(quotas: Dict[str, int], total: int) -> Dict[str, int]:
    """Largest-remainder scaling of `quotas` so that they sum to `total`."""
    s = sum(quotas.values())
    raw = {k: v * total / s for k, v in quotas.items()}
    out = {k: int(x) for k, x in raw.items()}
    rest = total - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[:rest]:
        out[k] += 1
    return out


@dataclass(frozen=True)
class Profile:
    name: str
    node_counts: Dict[str, int]
    edge_quotas: Dict[str, int]
    n_families: int
    n_campaigns: int
    path_counts: Tuple[int, int, int] = MANUSCRIPT_PATH_COUNTS
    n_expert: int = MANUSCRIPT_EXPERT_NODES
    class_counts: Dict[str, int] = field(default_factory=lambda: dict(MANUSCRIPT_CLASS_COUNTS))
    min_label_support: int = 3
    # fraction of edges that must stay inside a campaign / inside a campaign family (validation thresholds)
    min_campaign_local: float = 0.55
    min_family_local: float = 0.90

    @property
    def n_nodes(self) -> int:
        return sum(self.node_counts.values())

    @property
    def n_edges(self) -> int:
        return sum(self.edge_quotas.values())


_DEV_NODES = {"threat_actor": 24, "vulnerability": 210, "attack_method": 12, "file": 30,
              "attack_type": 24, "device": 32, "platform": 18}
_DEV_EDGES = round(MANUSCRIPT_EDGES * sum(_DEV_NODES.values()) / MANUSCRIPT_NODES)

PROFILES: Dict[str, Profile] = {
    "dev": Profile("dev", _DEV_NODES, scale_quotas(MANUSCRIPT_EDGE_QUOTAS, _DEV_EDGES),
                   n_families=4, n_campaigns=12, n_expert=round(MANUSCRIPT_EXPERT_NODES * 350 / MANUSCRIPT_NODES),
                   min_label_support=2, min_family_local=0.85),
    "manuscript_scale": Profile("manuscript_scale", dict(MANUSCRIPT_NODE_COUNTS), dict(MANUSCRIPT_EDGE_QUOTAS),
                                n_families=10, n_campaigns=60),
}
