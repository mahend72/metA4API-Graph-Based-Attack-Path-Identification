# PROTOCOL FREEZE (nothing here has been run on final data)

Protocol-status: FROZEN
Open-items: real_data_feature_exclusion_list

Status: **frozen except the real-data feature-exclusion list** (needs the real label definitions, which are not reconstructed;
it is not invented here). Machine-readable copy: `iocevaluator/protocol.py::FROZEN`; `final_experiment_preflight` checks that a run's
configuration equals it. Before the final run also record the git commit and the `manifest.json` content hash.
`manuscript.tex` is not modified by anything in this repository.

Source column:  **M** = stated in `manuscript.tex`;  **I** = implementation choice because the manuscript is underspecified
(each one is open to your veto);  **D** = library / generator default that the manuscript does not mention;
**OPEN** = needs a decision before the freeze.

No value in this file was chosen to move a result towards a manuscript number. Budget and search settings were chosen from
convergence/feasibility on development data only (§9). Manuscript results are historical reference values.

## 1. Dataset / profile
| Item | Setting | Src |
|---|---|---|
| Final data | the real AM-focused CTI TIKG, reconstructed and checksummed **before** the freeze (`provenance: real`) | M (3,728 nodes, 11,624 relations, 3,614 labelled stated) |
| Development data | `synthetic` and `semi_synthetic` `dev` profiles (350 nodes, 76 reference paths); development / testing only | I |
| Scale profile for cost measurements | `manuscript_scale` synthetic, 3,728 nodes, 11,624 edges, generator seed 42 | I (size and edge count from M) |
| Provenance rule | synthetic → development only; semi_synthetic → real measurements on a benchmark, not a reproduction; real → eligible for the manuscript | I |

## 2. Folds and seeds
| Item | Setting | Src |
|---|---|---|
| Outer protocol | 10-fold group-stratified CV, campaign-level isolation | M |
| Group | campaign; every entity of a campaign in one fold | M |
| Stratification | multi-label, by each node's rarest label | I |
| Outer split seed | `fold_seed = 0`, shared by every model and ablation variant | I |
| Inner validation | 10 % carve-out of the outer-train partition, campaign-isolated, never from the test fold (M allows 3-fold inner CV or a 10 % carve-out) | M / I |
| Model seeds | `0, 1, 2, 3, 4` → 10 × 5 = 50 runs per model | M (five initialisations); the seed values are I |
| Leakage | train-only scalers, χ statistics, MeGiTS similarities; test nodes isolated from the training graph; `verify_fold` raises on any overlap | M |
| Fallback | if a label class has < 10 members `n_splits` drops (only on tiny graphs); the manifest records `n_splits_used`. On the real data `n_splits_used` must equal 10, otherwise stop and report | I |

## 3. χ1–χ20 and MeGiTS weights
Definitions: `docs/CHI_DEFINITIONS.md` (source of truth: Fig. 3 topology, relation names from Table 2). Weights: uniform
`w_k = 1/20` over all 20 structures (M, Sec. 3.4); subsets renormalise to `1/|S|` (I). Per-structure similarity
`2·C_ij / (C_ii + C_jj)`; adjacency = Σ_k w_k · sim_k (M, Eq. 1). Four χ definitions (χ9 least certain; χ12, χ14, χ15) were derived from the
figure and are recorded as such in that file (I). Manuscript-text vs figure conflicts are listed there and are **not** resolved
by code. χ matrices and similarities are recomputed per fold from the outer-train graph (M).

## 4. Neural models (identical features, folds, optimiser, early stopping for all three — M, Sec. 4)
| Setting | GCN (reference) | GAT | HGT-style |
|---|---|---|---|
| Layers | 2 (M) | 2 (I) | 2 (I) |
| Hidden | 64 (M) | 64 = 4 heads × 16 (I) | 16 (I, parameter-budget width; `hgt_h64` sensitivity only) |
| Graph | MeGiTS adjacency, `D~^-1/2 (A+I) D~^-1/2` (M) | same MeGiTS support (I) | native TIKG types / relations, no MeGiTS edges (I) |
| Dropout | 0.5 (M) | 0.5 (I) | 0.5 (I) |
| Optimiser | Adam, lr 1e-3, weight decay 5e-4 (M) | same (M: "same optimisation settings") | same |
| Early stopping | validation Macro-F1, patience 20, max 500 epochs (M patience; max epochs I) | same | same |
| Output | per-label sigmoid, threshold 0.5, only labels valid for the entity type (I) | same | same |
Feature builder: default `FeatureConfig` (`text_backend="none"`, top-30 locations, top-20 organisations, `seed=7`) with the
target-defining fields excluded per the manuscript's feature–target separation (M); the exclusion list for the real data must
be written into the manifest before the freeze (**OPEN**).

## 5. External baselines
| | AttacKG (adaptation) | LADDER (step 3 adaptation) |
|---|---|---|
| Fit data | outer-train labels only; test campaign never visible | same |
| Key settings | γ = 0.5, node threshold 0.5, 2 hops, max graph nodes 40, max template instances 200, char-bigram Dice, 2^18 hashed bigram features | title weight 0.5, TF-IDF (1,2)-grams, nearest-description pooling |
| Threshold | selected on the validation partition from a 0.05–0.95 grid (I) | τ from a 0.05–1.00 grid on validation (I) |
| Seeds | seed-invariant; evaluated on the same (fold, seed) grid | same |
| Unsupported metrics | top-k and path metrics recorded as N/A with the reason | same |
Both are adaptations of the published descriptions to node-level IOC labelling (all components marked adapted / not reproduced
in `iocevaluator/cti_baselines.py::UNSUPPORTED_REASONS`); all values above are **I**.

## 6. Ranking: α, τ, EC (M, Sec. 3.6, with these gaps)
| Item | Setting | Src |
|---|---|---|
| Score | `R = α·EC + (1−α)·Severity` | M |
| Eigenvector centrality | principal eigenvector of `A_rank` (Adj without self-loops), max-normalised, computed within entity type | M (eigenvector); scope/normalisation I |
| Severity | CVSS for vulnerabilities; missing severity → EC only | M (CVSS) / I (missing rule) |
| α | **0.5, fixed** (primary). M says "tuned on the validation split" without a value; fixing α is a deviation (I). Tuned α (`alpha=None`) is a sensitivity configuration only | I |
| τ | **None** (primary): weighted ranking adjacency, no binarisation. M defines `A_bar = 1{Adj ≥ τ}` with τ "chosen on the validation split"; using no τ is a deviation (I). Binarised τ ∈ {0.05, 0.10, 0.25, 0.50} are sensitivity configurations only | I |
| EC scope | **type** (eigenvector centrality within each entity type; `ec_norm` max, missing severity → EC only) | I |
| Sensitivity | `protocol.sensitivity_ranker_configs()`: α ∈ {0, 0.25, 0.75, 1}, α tuned on validation, τ ∈ {0.05, 0.10, 0.25, 0.50}; each varies one setting; reported separately, fixed a priori, never part of the primary experiment, never chosen by performance | I |

## 7. Attack-path tracing (Algorithm 2)
| Item | Setting | Src |
|---|---|---|
| Seeds | nodes of the case's campaign with GCN confidence ≥ **0.5** (max predicted probability among labels valid for the type) | M ("high-confidence predicted nodes"); the value is I |
| Candidates | simple paths between two different seeds, 2…**6** edges, every hop a typed step of some χ template (`support="step"`) | M (conform to χ); length bounds and step-support I |
| Path-length limit | **6** — M groups references as 2, 3, ≥4 edges with no upper bound; 6 covers every reference length in the development data and keeps exhaustive search feasible (§9) | I |
| Search budget | **exhaustive**: no per-seed path budget; per-seed expansion *safety ceiling* 5,000,000 that must never be hit (`search_budget_hit` is recorded per case; **any hit raises `PathIntegrityError` for that case/run** (`EvalConfig.enforce_path_integrity`, default on) and the preflight re-checks every result row) | I |
| Retained candidates | top **5,000** by score. This cap is applied after ranking; it cannot change Hit@5 or Edge-F1 (tested) | I |
| Orientation | each path read from its higher-priority endpoint (larger R; ties smaller id) | I |
| Score | `Σ λ_c·component_c / Σ λ_c`, λ = 1 for priority, confidence, structure_weight; 0 for coherence; components in [0,1] | M names ingredients (MeGiTS weights, eigenvector centrality); formula I |
| Ranking | descending score rounded to 10 decimals; ties by node-id, relation and direction sequence | I |

## 8. Metrics
- Macro-F1, Micro-F1 (primary, M); ROC-AUC, PR-AUC (M); thresholds 0.5; only type-valid (node, label) pairs; Macro over labels with ≥ 1 positive in the scored nodes (I; `macro_f1_all_labels` also reported).
- Per-entity-type metrics (I). Top-k hit rate `1/Q Σ 1{R_q⁺ ∩ R_{q,k} ≠ ∅}` on the type-restricted campaign ranking, k ∈ {5, 10} (M definition; k values I).
- Exact-Path Hit@5: reference equals one of the 5 top-ranked candidates in ordered nodes, relations and edge directions (M text; strictness I). `exact_hit_either` (reversed reading) reported as a diagnostic only.
- Edge-F1: F1 of the stored-direction typed-edge sets of the top-ranked candidate vs the reference (M says "overlap between relations"; typed-edge reading I); `relation_f1` (relation multiset) reported as the looser reading.
- Aggregation: mean ± std over the 10 × 5 runs (M), plus fold-level t and bootstrap intervals (I).

## 9. Path-search evidence (development data) — why the search is exhaustive
Budget grid: candidate cap 5k/10k/25k/50k/∞ with per-seed paths and expansions scaled by the same factor (2,000/50,000 per seed at ×1),
× path-length limit {4, 5, 6}, 76 cases on both dev datasets (OOF GCN inputs, one model seed). Fixed convergence rule (written before
the runs): a level is acceptable if no search budget was hit, or top-5 lists equal the most complete level in ≥ 99 % of cases and
|ΔHit@5|, |ΔEdge-F1| ≤ 0.01. The rule never reads the metric level. Result (limit 6): with the earlier ×1 budgets 83 % (semi-synthetic) / 88 % (synthetic) of cases hit the per-seed budget and the top-5 list differs from the exhaustive result in 41 % / 62 % of cases.
On the unplanted dev inputs Hit@5 is 0 at every setting (near-chance model), so metric stability there is uninformative and the decision rests on top-5-list identity;
in the planting stress test only the uncapped level passes the rule in both datasets. A reference-planting stress test (reference nodes get maximal priority/confidence; development only, not a performance result)
shows Hit@5 would be **under-reported** by truncation (0.30 at ×1 vs 0.70 exhaustive, limit 6). Exhaustive search at 3,728 nodes
is feasible (§10 of the report): no case hit the safety ceiling.

## 10. Ablations (all vs `megits_full`, same folds / seeds)
`original_adjacency`; `binary_semantic`; `megits_paths_only` (χ1–11); `megits_graphs_only` (χ12–20); `megits_full` (reference); `chi1`…`chi20` each alone;
depth `megits_full_{1,3,4}layer`; label fraction `megits_full_lf{25,50,75}`. M names the four semantic variants and 1–4 layers; per-structure and
label-fraction variants follow Table 13 / the robustness text (I details).

## 11. Statistical tests
Paired Wilcoxon signed-rank on per-fold mean differences, α = 0.05 (M: "over per-fold scores"), vs the reference model aligned on (fold, seed) pairs
(alignment mismatch is an error), with **Holm** step-down adjustment per metric over the compared methods/variants (I — not in the manuscript). Baselines
vs the GCN use the same procedure (AttacKG, LADDER, GAT, HGT).

## 12. Timing methodology (cost tables)
`time.perf_counter`; `gc.collect()` before each timed call; untimed warm-up (1 call; 1 for training/baselines); median of ≥ 3 repeats (5 fast, 3 slow) with
mean/std/min/max/quartiles kept; components timed separately and split into one-time preprocessing vs algorithmic runtime; per-node latency =
whole-graph forward pass ÷ nodes (the manuscript's "newly inserted node batch" latency is **not** measured this way — recorded limitation);
memory = `tracemalloc` peak in a separate extra call (+ sampled RSS, lower bound). Timings are labelled by provenance and never mixed with
performance tables. Final cost numbers must be re-measured on the final machine; the manuscript's TPU timings are not comparable (M states a TPU v3-8).

## 13. Final-experiment preflight (`protocol.final_experiment_preflight`)
Fails unless: protocol marked frozen and only `real_data_feature_exclusion_list` open; ranker α not None; dataset provenance known;
a feature-exclusion policy supplied **and** applied to `FeatureConfig`; folds/seeds/fold_seed/val_fraction equal the freeze (no `max_folds`);
χ1–χ20 weights uniform 1/20; ranking and path settings equal `FROZEN`; path integrity enforcement on; (post-run) no `search_budget_hit` row;
git commit known, clean tree (dry runs may pass `allow_dirty`), and protocol / config / code hashes recorded in the report.
`EvalConfig` defaults are now `paths=exhaustive_path_config()` and `enforce_path_integrity=True`.

## 14. Still open
1. **Real-data feature-exclusion list** (OPEN; left empty on purpose until the real label definitions are reconstructed).
2. Whether the final path evaluation traces all 76 reference cases for every (fold, seed) (≈ 380 traces, ≈ 5–7 s each at 3,728 nodes).
3. Commit hash + manifest content hash to record at the final run (the preflight records them).
