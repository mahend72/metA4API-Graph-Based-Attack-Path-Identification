"""Node feature matrix F (manuscript Sec. 3.4.5, Table 3) for every TIKG node.

Features (all computed from `Entity.attrs`; missing values -> 0 plus a `has_*` indicator, never imputed):
  active time (last_seen - first_seen), recency, update frequency, source/target location multi-hot,
  organisation/industry multi-hot, entity-type (and file-subtype) one-hot, optional text representation.

Leakage control (Sec. 3.4.5 / 4.2):
  * `exclude_fields`: attrs keys removed before featurisation (e.g. the fields that define a label such as
    "cwe", "cvss", "attack_id").
  * `strip_terms`: strings (e.g. label class names) removed from text before the text representation is built.
  * all statistics (scaler, vocabularies, SVD, recency reference time) are fit on the training nodes only.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence

import numpy as np

from .tikg import TIKG, EntityType, FILE_SUBTYPES


def _ts(v) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).replace(tzinfo=None).timestamp() / 86400.0
    except ValueError:
        return None


@dataclass
class FeatureConfig:
    exclude_fields: Sequence[str] = ()
    strip_terms: Sequence[str] = ()
    top_locations: int = 30
    top_orgs: int = 20
    text_backend: str = "none"       # none | tfidf | sbert
    text_dim: int = 32
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    seed: int = 7


class FeatureBuilder:
    def __init__(self, cfg: FeatureConfig | None = None):
        self.cfg = cfg or FeatureConfig()
        self.names: List[str] = []

    def _attr(self, e, key):
        return None if key in self.cfg.exclude_fields else e.attrs.get(key)

    def _text(self, tikg: TIKG) -> List[str]:
        pats = [re.compile(re.escape(t), re.I) for t in self.cfg.strip_terms if t]
        out = []
        for e in tikg.entities:
            txt = " ".join(str(x) for x in (e.name, self._attr(e, "text") or "", e.type.value) if x)
            for p in pats:
                txt = p.sub(" ", txt)
            out.append(txt)
        return out

    def _raw(self, tikg: TIKG):
        n = tikg.n
        first = np.array([_ts(self._attr(e, "first_seen")) or np.nan for e in tikg.entities])
        last = np.array([_ts(self._attr(e, "last_seen")) or np.nan for e in tikg.entities])
        upd = np.array([float(self._attr(e, "update_count") or 0) for e in tikg.entities])
        return first, last, upd

    def fit(self, tikg: TIKG, train_idx: np.ndarray) -> "FeatureBuilder":
        c = self.cfg
        first, last, upd = self._raw(tikg)
        tr = np.asarray(train_idx)
        self._ref = np.nanmax(last[tr]) if np.isfinite(last[tr]).any() else 0.0
        loc = Counter(l for i in tr for k in ("source_countries", "target_countries")
                      for l in (self._attr(tikg.entities[i], k) or []))
        org = Counter(o for i in tr for o in (self._attr(tikg.entities[i], "industries") or []))
        self._locs = [k for k, _ in loc.most_common(c.top_locations)]
        self._orgs = [k for k, _ in org.most_common(c.top_orgs)]
        num = self._numeric(tikg, first, last, upd)
        self._mu = num[tr].mean(0)
        self._sd = num[tr].std(0)
        self._sd[self._sd < 1e-8] = 1.0
        self._text_model = None
        if c.text_backend == "tfidf":
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            txt = self._text(tikg)
            self._tfidf = TfidfVectorizer(min_df=1, max_features=5000).fit([txt[i] for i in tr])
            X = self._tfidf.transform([txt[i] for i in tr])
            k = max(1, min(c.text_dim, X.shape[1] - 1, X.shape[0] - 1))
            self._svd = TruncatedSVD(k, random_state=c.seed).fit(X)
        return self

    def _numeric(self, tikg, first, last, upd) -> np.ndarray:
        active = np.where(np.isfinite(first) & np.isfinite(last), last - first, 0.0)
        recency = np.where(np.isfinite(last), self._ref - last, 0.0)
        has_t = (np.isfinite(first) & np.isfinite(last)).astype(float)
        return np.c_[np.log1p(np.maximum(active, 0)), np.log1p(np.maximum(recency, 0)), np.log1p(upd), has_t]

    def transform(self, tikg: TIKG) -> np.ndarray:
        first, last, upd = self._raw(tikg)
        num = (self._numeric(tikg, first, last, upd) - self._mu) / self._sd
        blocks = [num]
        names = ["active_time", "recency", "update_frequency", "has_time"]
        loc = np.zeros((tikg.n, 2 * len(self._locs)))
        org = np.zeros((tikg.n, len(self._orgs)))
        li = {l: j for j, l in enumerate(self._locs)}
        oi = {o: j for j, o in enumerate(self._orgs)}
        for i, e in enumerate(tikg.entities):
            for side, key in enumerate(("source_countries", "target_countries")):
                for l in self._attr(e, key) or []:
                    if l in li:
                        loc[i, side * len(self._locs) + li[l]] = 1
            for o in self._attr(e, "industries") or []:
                if o in oi:
                    org[i, oi[o]] = 1
        names += [f"src_loc:{l}" for l in self._locs] + [f"tgt_loc:{l}" for l in self._locs]
        names += [f"org:{o}" for o in self._orgs]
        et = np.array([[t.value == e.type.value for t in EntityType] for e in tikg.entities], dtype=float)
        st = np.array([[e.subtype == s for s in FILE_SUBTYPES] for e in tikg.entities], dtype=float)
        names += [f"type:{t.value}" for t in EntityType] + [f"subtype:{s}" for s in FILE_SUBTYPES]
        blocks += [loc, org, et, st]
        if self.cfg.text_backend == "tfidf":
            txt = self._text(tikg)
            z = self._svd.transform(self._tfidf.transform(txt))
            blocks.append(z)
            names += [f"text{j}" for j in range(z.shape[1])]
        elif self.cfg.text_backend == "sbert":
            from .features import build_text_embeddings
            z = build_text_embeddings(self._text(tikg), self.cfg.sbert_model)
            blocks.append(z)
            names += [f"sbert{j}" for j in range(z.shape[1])]
        self.names = names
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def fit_transform(self, tikg: TIKG, train_idx: np.ndarray) -> np.ndarray:
        return self.fit(tikg, train_idx).transform(tikg)
