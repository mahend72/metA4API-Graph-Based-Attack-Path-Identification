"""SYNTHETIC toy TIKG used ONLY by the unit tests. It is random, has no relation to real CTI, and must never be
used to report results."""
from __future__ import annotations

import numpy as np

from iocevaluator.tikg import TIKG, Entity, EntityType as ET


def make_synthetic(n_campaigns: int = 12, seed: int = 0):
    rng = np.random.RandomState(seed)
    ents, trips, labels = [], [], {}

    def add(eid, t, camp, fam, subtype="", name=None):
        ents.append(Entity(eid, t, name or eid, subtype, camp,
                           {"industries": [f"fam{fam}"], "update_count": int(rng.randint(1, 5)),
                            "first_seen": "2024-01-01T00:00:00", "last_seen": f"2024-0{1 + fam}-15T00:00:00",
                            "target_countries": ["GB" if fam == 0 else "IN"]}))
        labels[eid] = [f"fam{fam}"]
        return eid

    plats = [add(f"plat{i}", ET.PLATFORM, f"c{i}", i % 3) for i in range(3)]
    for c in range(n_campaigns):
        fam, camp = c % 3, f"c{c}"
        tas = [add(f"ta{c}_{j}", ET.THREAT_ACTOR, camp, fam) for j in range(2)]
        vs = [add(f"cve{c}_{j}", ET.VULNERABILITY, camp, fam) for j in range(2)]
        m = add(f"m{c}", ET.ATTACK_METHOD, camp, fam)
        at = add(f"at{c}", ET.ATTACK_TYPE, camp, fam)
        ds = [add(f"dev{c}_{j}", ET.DEVICE, camp, fam) for j in range(2)]
        fs = [add(f"f{c}_hash", ET.FILE, camp, fam, "hash"), add(f"f{c}_dom", ET.FILE, camp, fam, "domain"),
              add(f"f{c}_ip", ET.FILE, camp, fam, "ip")]
        for ta in tas:
            trips += [(ta, "exploits", vs[0]), (ta, "uses", m)] + [(ta, "uses", f) for f in fs]
        trips += [(tas[0], "unauthorised_access", ds[0]), (tas[1], "unauthorised_access", ds[1]),
                  (tas[0], "assists", tas[1]), (m, "exploits", vs[0]), (m, "exploits", vs[1]),
                  (at, "exploits", vs[0]), (vs[0], "evolves_to", vs[1]), (fs[0], "contains", vs[0])]
        for d in ds:
            trips += [(d, "affected_by", vs[0]), (d, "runs_on", plats[fam])]
        trips.append((plats[fam], "affected_by", vs[0]))
    return TIKG(ents, trips), labels
