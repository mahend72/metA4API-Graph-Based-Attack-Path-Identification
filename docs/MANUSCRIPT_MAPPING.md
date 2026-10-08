# Manuscript -> code map and open inconsistencies

## Map
| Manuscript | Code |
|---|---|
| Sec. 3.2 TIKG, Tables 1-2, 9 relations | `iocevaluator/tikg.py` |
| Sec. 3.3 chi_1..chi_20, Alg. 1 | `iocevaluator/metagraphs.py` |
| Eq. 1-3 MeGiTS, weights, per-fold leakage control | `iocevaluator/megits.py` |
| Sec. 3.4.5 features, Table 3 | `iocevaluator/tikg_features.py` |
| Eq. 4-5 GCN / BCE, GCN depth, GAT, HGT-style | `iocevaluator/models.py` |
| Sec. 3.6 EC, A_rank, binarised A_rank, fused risk, Table 19 explanations | `iocevaluator/prioritisation.py` |
| Alg. 2 path tracing / ranking, path-level metrics | `iocevaluator/attack_paths.py` |
| Sec. 4.5 CV protocol | `iocevaluator/splits.py`, `experiments.py` |
| Metrics, Wilcoxon | `iocevaluator/metrics.py` |
| Tables 6, 8-10, 12-15 experiments | `experiments.py` variants + `iocevaluator evaluate` |

## Inconsistencies found in `manuscript.tex` (manuscript NOT modified)
1. **chi definitions.** chi_1..chi_20 are implemented from the node topology of `fig3.png` (see `docs/CHI_DEFINITIONS.md`);
   a few Fig. 3 edge labels (chi_2, 6, 7, 9) and one node label (chi_10 "AM") are inconsistent with Table 2 and are read
   as the Table 2 triplet with the same node types.
2. **chi_13 / chi_18 prose.** Sec. 3.3 describes chi_13 as actors sharing file hashes/e-mails and chi_18 as actors /
   domains / IPs / e-mails; Fig. 3 and Table 13 give chi_13 = D=>{V,P}=>D and chi_18 = TA-M-V-M-TA. Fig. 3 is implemented.
3. **Table 13 "and" is not one operator.** chi_12/13/14/20 are Hadamard products, chi_15..chi_18 are `Q C Q^T` chains (as in
   Algorithm 1), chi_19 combines both; the per-structure operator is only visible in Fig. 3.
4. **No actor->file triplet.** Not needed by any structure once Fig. 3 is used; the optional synthetic extension
   `X1 = <TA, uses, F>` is not used by chi_1..chi_20.
5. **`enables` has no triplet** in Table 2 (relation is defined, never used).
6. **Symbol clash:** "D" is Device in Tables 2 / Alg. 1 but Domain in the chi_18 text; "A" vs "TA" for actors.
7. **MeGiTS is same-type only.** Eq. 1 needs NumP(v,v), i.e. symmetric structures, so the MeGiTS adjacency is
   block-diagonal by entity type; no cross-type message passing happens in the GCN (only via features). The text's
   claim that it captures "long-range" dependencies across types should be qualified.
8. **Table 13 vs Tables 8-10.** chi_19 alone reports exactly the full-model scores (0.7551 / 0.7696) although the
   full model is described as uniform over all 20 structures. Either the full model is chi_19-only or the row is copied.
9. **Label coverage:** 3,614 + 114 = 3,728 = 100% of nodes, yet the text says 96.9% are labelled.
10. **Table 7 vs 10-fold CV.** Table 7 gives one fixed train/val/test split (about 89/5.5/5.5%) while all results use
    10-fold CV with 10% validation carve-out.
11. **Wrong cross-references:** the GCN-depth comparison is cited as Sec. `sec4.5` (that label is the evaluation
    protocol; depth is `sec4.3`), and "the protocol described in Section `sec4.2`" actually lives in `sec4.5`.
12. **Undefined "label hierarchy"** (Macro-F1 definition, Eq. 5) - labels are flat multi-label here.
13. **Path score, seed rule, "high-risk" definition, tau/alpha tuning criterion, campaign definition, how
    "case scope" is given for path evaluation** are unspecified; implemented as explicit configurable choices
    (documented in `attack_paths.py`, `prioritisation.py`).
14. **PyTorch vs TensorFlow/TPU.** The manuscript (Sec. 4.2) states TensorFlow 2.x with XLA/TPUStrategy on a TPU v3-8;
    the repo is PyTorch (CPU). Same model/loss/optimiser, but numbers need not be bit-identical. Not silently changed:
    `manifest.json` records the backend.
15. **Early stopping.** "Patience 20 on validation Macro-F1": ties are broken by validation loss, otherwise the
    counter expires at F1 = 0 before learning begins.
16. **No results are reproduced or claimed.** The numbers in Tables 6-20 require the unavailable corpus and labels.
