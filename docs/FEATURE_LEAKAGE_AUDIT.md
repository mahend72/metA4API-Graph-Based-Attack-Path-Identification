# Feature-leakage audit (semi-synthetic, `manuscript_scale`, 3,728 nodes)

Scope: dataset JSON → `datasets.load_dataset` → `tikg_from_dict` → `FeatureConfig`/`FeatureBuilder` → `prepare_fold` → model input `x`.
Policy produced from it: `configs/feature_policy_semi_synthetic.json` (semi-synthetic / synthetic only; the real-data policy is still OPEN).

## How the model matrix is built
* `FeatureBuilder` is an **allow-list**. It reads only `attrs["first_seen"]`, `["last_seen"]`, `["update_count"]`,
  `["source_countries"]`, `["target_countries"]`, `["industries"]` and, **only if `text_backend != "none"`** (default `none`), `["text"]` plus `Entity.name` and the type.
  It also uses the entity type and file subtype (one-hot). Nothing else in `attrs` is ever read.
* All statistics (scaler mean/sd, recency reference time, location/industry vocabularies, TF-IDF vocabulary/SVD) are fitted on the **outer-train nodes only**
  (`prepare_fold`: `fb.fit_transform(tikg, flatnonzero(masks.train))`; validation is excluded too).
* The GCN receives `x` and the MeGiTS adjacency (graph structure, no labels). Labels never enter `x`.
* CVSS is loaded into a separate `severity` array used **only** by the prioritisation ranker (Severity term). CWE is not consumed anywhere in the package.

## Per-field decisions
| Field (source) | Entity types | Used by builder | Fitted on train? | Can encode | Decision |
|---|---|---|---|---|---|
| `first_seen`, `last_seen` → active_time, recency, has_time | all (vuln: CVE published / last-modified) | yes | scaler + recency reference: train only | campaign timing; probed: no cross-campaign label signal (≈ majority rate) | **include** |
| `update_count` | all except vulnerability | yes | scaler: train only | none found (probe ≈ majority) | **include** |
| entity type, file subtype (one-hot) | all | yes | no fitting | the label *space* is per type, so type is not a label proxy | **include** |
| `source_countries`, `target_countries` | all except vulnerability | yes (multi-hot) | vocabulary: train only | **family identity**: campaign-level values, shared by related campaigns of a family; labels are drawn from a family-anchored distribution. Probe (cross-campaign, logistic, 5-fold group CV): device 0.42 vs majority 0.09, attack_method 0.68 vs 0.43, platform 0.59 vs 0.14 | **exclude** |
| `industries` | all except vulnerability | yes (multi-hot) | vocabulary: train only | same mechanism; 0.32–0.51 vs 0.09–0.14 | **exclude** |
| `text` | all | only with text backend | TF-IDF/SVD: train only | family theme phrases ("robotic cell and controller attack"), role words | **exclude**; keep `text_backend="none"` |
| `Entity.name` | all | only inside text | – | generated names/device kinds are family-flavoured | unused while text backend is `none` |
| `family` | non-vulnerability | no | – | latent generator family; labels anchored on it | **exclude** (defensive) |
| `Entity.campaign` / `campaign_id`, `family_id` | all | no | – | fold membership / campaign identity (the CV group key) | **exclude** |
| `synthetic`, `provenance`, `synthetic_fields`, `cve_snapshot_sha256` | all / vulnerability | no | – | generator provenance; separates real-CVE nodes from fabricated ones | **exclude** |
| `cvss_base_score`, `cvss_vector`, `cvss_version`, `severity` | vulnerability | no | – | CVE-to-campaign assignment is family-driven, so severity correlates with family | **exclude from features; ranking-only metadata** (severity array) |
| `cwe` | vulnerability | no | – | weakness vocabulary is family-specific in the assignment | **exclude**; not used anywhere |
| `alias`, `technique_id`, `vendor`, `model`, `device_kind`, `platform_kind`, `version`, `affected_version`, `hash_algorithm`, `file_name`, `file_size`, `vendors`, `products`, `cpe`, `platform_parts`, `cve_id`, `published`, `last_modified`, `vuln_status`, `source_identifier`, `references` | various | no | – | identifiers / family-mixed catalogue attributes | **exclude** (defensive) |
| labels / class codes | – | no | – | targets | never in `x`; test labels are zeroed in the visible label matrix |
| reference paths (`attack_paths.json`) | – | no | – | evaluation only | never in `x`, adjacency or training |
| node ids | – | no | – | e.g. `actor:SYN-TA-0271`, `vuln:CVE-…` | never read |

## Interpretation caveat (structure, not features)
The generator also biases **edges** toward same-campaign / related-campaign / other-family targets (`generator.py`, edge locality). Graph structure therefore carries
family signal by design; this is the method under test (MeGiTS adjacency), not a feature leak, but it means semi-synthetic scores measure the pipeline on a generator whose
structure is informative. Excluding country/industry removes the *attribute* shortcut only.

## Enforced by tests (`tests/test_feature_leakage.py`)
Policy completeness and per-field reasons; real data rejected for this policy; keys actually read ⊆ {first_seen, last_seen, update_count} under the policy;
label-perfect values injected into every forbidden field (and `Entity.campaign`, names) leave the matrix byte-identical; the same injection changes it without the policy;
validation/test attribute mutations leave scaler, vocabularies, TF-IDF vocabulary and all train rows unchanged.
