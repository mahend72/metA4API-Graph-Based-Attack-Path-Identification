from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .schemas import Event

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelationMatrices:
    attackers: List[str]
    file_hash: List[str]
    email: List[str]
    ip: List[str]
    url: List[str]
    domain: List[str]
    hostname: List[str]
    region: List[str]
    attack_technique: List[str]
    vulnerability: List[str]

    # Each relation is attacker x type
    R_file_hash: np.ndarray
    R_email: np.ndarray
    R_ip: np.ndarray
    R_url: np.ndarray
    R_domain: np.ndarray
    R_hostname: np.ndarray
    R_region: np.ndarray
    R_attack_technique: np.ndarray
    R_vulnerability: np.ndarray


def _unique(values: Iterable[str]) -> List[str]:
    return sorted({v for v in values if v})


def build_relations(events: List[Event]) -> RelationMatrices:
    attackers = _unique(e.attacker for e in events)
    file_hash = _unique(v for e in events for v in e.file_hash)
    email = _unique(v for e in events for v in e.email)
    ip = _unique(v for e in events for v in e.ip)
    url = _unique(v for e in events for v in e.url)
    domain = _unique(v for e in events for v in e.domain)
    hostname = _unique(v for e in events for v in e.hostname)
    region = _unique(v for e in events for v in e.region)
    attack_technique = _unique(v for e in events for v in e.attack_technique)
    vulnerability = _unique(v for e in events for v in e.vulnerability)

    idxA = {a: i for i, a in enumerate(attackers)}

    def mat(cols: List[str]) -> tuple[np.ndarray, Dict[str, int]]:
        return np.zeros((len(attackers), len(cols)), dtype=np.float32), {c: j for j, c in enumerate(cols)}

    R_file_hash, idxH = mat(file_hash)
    R_email, idxE = mat(email)
    R_ip, idxIP = mat(ip)
    R_url, idxU = mat(url)
    R_domain, idxD = mat(domain)
    R_hostname, idxHost = mat(hostname)
    R_region, idxR = mat(region)
    R_attack, idxAT = mat(attack_technique)
    R_vuln, idxV = mat(vulnerability)

    for e in events:
        ai = idxA[e.attacker]
        for v in e.file_hash:
            R_file_hash[ai, idxH[v]] = 1.0
        for v in e.email:
            R_email[ai, idxE[v]] = 1.0
        for v in e.ip:
            R_ip[ai, idxIP[v]] = 1.0
        for v in e.url:
            R_url[ai, idxU[v]] = 1.0
        for v in e.domain:
            R_domain[ai, idxD[v]] = 1.0
        for v in e.hostname:
            R_hostname[ai, idxHost[v]] = 1.0
        for v in e.region:
            R_region[ai, idxR[v]] = 1.0
        for v in e.attack_technique:
            R_attack[ai, idxAT[v]] = 1.0
        for v in e.vulnerability:
            R_vuln[ai, idxV[v]] = 1.0

    log.info(
        "Relation matrices built: attackers=%d | hash=%d email=%d ip=%d url=%d domain=%d host=%d region=%d at=%d cve=%d",
        len(attackers),
        len(file_hash),
        len(email),
        len(ip),
        len(url),
        len(domain),
        len(hostname),
        len(region),
        len(attack_technique),
        len(vulnerability),
    )

    return RelationMatrices(
        attackers=attackers,
        file_hash=file_hash,
        email=email,
        ip=ip,
        url=url,
        domain=domain,
        hostname=hostname,
        region=region,
        attack_technique=attack_technique,
        vulnerability=vulnerability,
        R_file_hash=R_file_hash,
        R_email=R_email,
        R_ip=R_ip,
        R_url=R_url,
        R_domain=R_domain,
        R_hostname=R_hostname,
        R_region=R_region,
        R_attack_technique=R_attack,
        R_vulnerability=R_vuln,
    )


def commuting_matrices(rel: RelationMatrices) -> Dict[str, np.ndarray]:
    """Compute attacker-attacker commuting matrices for each relation type.

    Equivalent to notebook cells like:
        M1 = R1 @ R1.T   (attacker-hash-attacker)
    """
    mats = {
        "hash": rel.R_file_hash @ rel.R_file_hash.T,
        "email": rel.R_email @ rel.R_email.T,
        "ip": rel.R_ip @ rel.R_ip.T,
        "url": rel.R_url @ rel.R_url.T,
        "domain": rel.R_domain @ rel.R_domain.T,
        "hostname": rel.R_hostname @ rel.R_hostname.T,
        "region": rel.R_region @ rel.R_region.T,
        "attack_technique": rel.R_attack_technique @ rel.R_attack_technique.T,
        "vulnerability": rel.R_vulnerability @ rel.R_vulnerability.T,
    }
    return mats
