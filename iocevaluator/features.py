from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureArtifacts:
    attackers: List[str]
    feature_names: List[str]
    X: np.ndarray  # shape [N, D]


def build_structured_attacker_features(
    threats: Iterable,
    attackers: List[str],
) -> pd.DataFrame:
    """Build simple structured features per attacker.

    - active_time_days: mean days since record created
    - update_frequency: mean revision
    - one-hot industries / targeted_countries (union across attacker records)
    """
    threats = list(threats)
    now = datetime.now()

    all_orgs = sorted({org for t in threats for org in (getattr(t, "industries", None) or []) if org})
    all_locs = sorted({loc for t in threats for loc in (getattr(t, "targeted_countries", None) or []) if loc})

    cols = ["active_time_days", "update_frequency"] + [f"org:{o}" for o in all_orgs] + [f"loc:{l}" for l in all_locs]
    df = pd.DataFrame(0.0, index=attackers, columns=cols)

    by_attacker: Dict[str, List] = {a: [] for a in attackers}
    for t in threats:
        a = (getattr(t, "adversary", "") or "").strip()
        if a in by_attacker:
            by_attacker[a].append(t)

    for a, recs in by_attacker.items():
        if not recs:
            continue
        active = [(now - r.created).days for r in recs]
        rev = [float(getattr(r, "revision", 0) or 0) for r in recs]
        df.loc[a, "active_time_days"] = float(np.mean(active))
        df.loc[a, "update_frequency"] = float(np.mean(rev))

        orgs = {o for r in recs for o in (getattr(r, "industries", None) or []) if o}
        locs = {l for r in recs for l in (getattr(r, "targeted_countries", None) or []) if l}
        for o in orgs:
            df.loc[a, f"org:{o}"] = 1.0
        for l in locs:
            df.loc[a, f"loc:{l}"] = 1.0

    return df


def build_text_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> np.ndarray:
    """Encode text to dense vectors using SentenceTransformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def build_attacker_feature_matrix(
    threats: Iterable,
    attackers: List[str],
    feature_dim: int = 32,
    use_text_embeddings: bool = True,
    text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    random_state: int = 7,
) -> FeatureArtifacts:
    """Create a dense numeric feature matrix X for attackers.

    Steps:
      1) structured features
      2) optional text embedding per attacker (from names + malware families + techniques)
      3) concatenate and (optionally) reduce to `feature_dim` using TruncatedSVD
      4) standardize to zero-mean, unit-variance
    """
    structured = build_structured_attacker_features(threats, attackers)
    blocks = [structured.values.astype(np.float32)]
    feature_names = list(structured.columns)

    if use_text_embeddings:
        threat_list = list(threats)
        by_att = {a: [] for a in attackers}
        for t in threat_list:
            a = (getattr(t, "adversary", "") or "").strip()
            if a in by_att:
                by_att[a].append(t)

        texts: List[str] = []
        for a in attackers:
            recs = by_att.get(a, [])
            parts = [a]
            for r in recs:
                parts.append(getattr(r, "name", "") or "")
                parts.extend(getattr(r, "malware_families", None) or [])
                parts.extend(getattr(r, "attack_ids", None) or [])
            texts.append(" ".join([p for p in parts if p]))

        emb = build_text_embeddings(texts, model_name=text_embedding_model)
        blocks.append(emb)
        feature_names.extend([f"emb:{i}" for i in range(emb.shape[1])])

    X = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)

    # Pad if fewer features than requested (keeps stable dimension for downstream models).
    if X.shape[1] < feature_dim:
        pad = np.zeros((X.shape[0], feature_dim - X.shape[1]), dtype=np.float32)
        X = np.concatenate([X, pad], axis=1)
        feature_names.extend([f"pad:{i}" for i in range(pad.shape[1])])

    # Reduce if more than requested.
    if X.shape[1] > feature_dim:
        from sklearn.decomposition import TruncatedSVD

        svd = TruncatedSVD(n_components=feature_dim, random_state=random_state)
        X = svd.fit_transform(X).astype(np.float32)
        feature_names = [f"svd:{i}" for i in range(feature_dim)]

    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(X).astype(np.float32)

    log.info("Built attacker features: N=%d, D=%d", X.shape[0], X.shape[1])
    return FeatureArtifacts(attackers=attackers, feature_names=feature_names, X=X)
