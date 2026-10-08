"""Campaign-isolated, label-stratified K-fold splits (manuscript Sec. 4.5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .labels import LabelSpace
from .tikg import TIKG


@dataclass
class Fold:
    index: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    test_nodes_mask: np.ndarray      # ALL nodes (labelled or not) belonging to a test campaign -> hidden from train graph


def group_stratified_folds(tikg: TIKG, labels: LabelSpace, n_splits: int = 10, seed: int = 0,
                           val_fraction: float = 0.10, allow_ungrouped: bool = False) -> List[Fold]:
    """Folds over labelled nodes. Every node of a campaign lands in the same fold (campaign-level isolation);
    multi-label stratification uses each node's rarest label. The validation set is carved out of the
    outer-train partition only, again group-isolated."""
    groups_all = tikg.campaigns()
    if (groups_all == "").any():
        if not allow_ungrouped:
            raise ValueError("entities without a `campaign` cannot be group-isolated; set campaigns or "
                             "pass allow_ungrouped=True (each node becomes its own group - NOT campaign isolation)")
        groups_all = np.array([g if g else f"__node{i}" for i, g in enumerate(groups_all)], dtype=object)
    lab = np.flatnonzero(labels.labelled)
    groups = groups_all[lab]
    if len(set(groups)) < n_splits:
        raise ValueError(f"only {len(set(groups))} campaigns for {n_splits} folds")
    strat = labels.primary_label()[lab]
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: List[Fold] = []
    for k, (tr, te) in enumerate(skf.split(lab, strat, groups)):
        tr_idx, te_idx = lab[tr], lab[te]
        if len(set(groups[tr])) > 1 and val_fraction > 0:
            gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed + k)
            a, b = next(gss.split(tr_idx, groups=groups[tr]))
            tr_idx, va_idx = tr_idx[a], tr[b]
            va_idx = lab[va_idx]
        else:
            va_idx = tr_idx[:0]
        test_campaigns = set(groups_all[te_idx])
        mask = np.array([g in test_campaigns for g in groups_all])
        folds.append(Fold(k, np.sort(tr_idx), np.sort(va_idx), np.sort(te_idx), mask))
    return folds
