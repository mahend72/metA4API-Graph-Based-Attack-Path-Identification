# Data required to reproduce the manuscript (NOT shipped, NOT fabricated)

The repository contains code only. Everything below must come from the authors' real corpus. Empty templates are in
`data/templates/`. `data/sample_threats.json` is a tiny format example, not research data.

| Need (manuscript section) | File / format | Status |
|---|---|---|
| AM-focused CTI corpus, 3,728 nodes / 11,624 relations (Sec. 4.1, Table 4) | OTX pulses -> `export_otx_to_json.py` -> `iocevaluator build-tikg` | **Required** (needs OTX key + the authors' AM/ICS filters; the curation steps - keyword + expert filtering - are not scripted anywhere) |
| Devices, platforms, attack types and relations T2-T10 | `--extra` TIKG JSON (`data/templates/tikg_extra.json`): `{"entities":[{id,type,name,subtype,campaign,attrs}], "triplets":[{s,r,t}]}` | **Required**; OTX contains none of them |
| Campaign ids for group isolation | `campaign` field on every entity (the OTX adapter uses the first pulse name) | **Required** for the 10-fold protocol; the manuscript does not say how campaigns were defined |
| Labels (3,614 public-CTI + 114 expert-annotated nodes, Table 7 class vocabularies) | `labels.json`: `{"labels": {entity_id: [class,...]}, "classes": {entity_type: [class,...]}}` | **Required** |
| CVSS / impact severity (Sec. 3.6) | `severity.json`: `{"scale": 10, "scores": {entity_id: score}}`; unknown nodes are *not* imputed | **Required** for fusion; without it R = EC |
| Reference attack paths (76 cases, Sec. 4.6) | `reference_paths.json`: `[{"case_id","campaign","nodes":[ids],"relations":[...],"case_nodes":[ids]?}]` | **Required** for Exact-Path Hit@5 / Edge-F1 |
| Expert-confirmed high-risk IOCs per alert case (top-k hit rate) | list of `(ranked_ids, confirmed_set)` passed to `metrics.topk_hit_rate` | **Required** |
| Validation ground truth to tune alpha and tau | callable passed to `prioritisation.tune_alpha` / `select_tau` | **Required** |
| AttacKG and LADDER predictions | N x K out-of-fold probability arrays -> `experiments.evaluate_external_oof` | **Blocked** (external systems; not re-implemented) |
| Fig. 3 (definition of chi_1..chi_20) | `iocevaluator/metagraphs.py`, `docs/CHI_DEFINITIONS.md` | **Done** - implemented from `fig3.png` |

Leakage control: pass the attrs fields that define labels via `--exclude-fields` (e.g. `cwe cvss attack_id`).
