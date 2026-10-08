# metA4API-Graph-Based-Attack-Path-Identification

A small, reproducible Python repo that turns threat-feed IOC data (e.g., AlienVault OTX pulses) into:

- **Event graph** (attacker ↔ IOC types)
- **Meta-path commuting matrices**
- **MIIS/MPGIIS-style attacker similarity matrix** (Dice on meta-path counts)
- Optional **GCN autoencoder embeddings** (unsupervised)

This repo is a cleaned, modularised implementation based on the linked `IOCEvaluator.ipynb` notebook.

## Repository structure

```text
.
├─ iocevaluator/                 # library code (importable package)
│  ├─ io_loader.py               # load/save normalised JSON
│  ├─ otx_client.py              # fetch + normalise OTX pulses (no keys in code)
│  ├─ event_builder.py           # build Event objects from ThreatRecords
│  ├─ relations.py               # attacker↔IOC relation matrices + commuting matrices
│  ├─ mpgiis.py                  # MIIS/MPGIIS-style aggregation
│  ├─ features.py                # attacker feature matrix builder
│  ├─ gcn.py                     # optional GCN autoencoder (PyTorch)
│  ├─ pipeline.py                # end-to-end pipeline
│  └─ cli.py                     # command line interface
├─ scripts/
│  └─ export_otx_to_json.py       # helper: export OTX to normalised JSON
├─ notebooks/
│  └─ IOCEvaluator_original.ipynb # your original notebook (kept as reference)
├─ .env.example
├─ pyproject.toml
└─ README.md
```

## Install

Create a virtual environment, then:

```bash
pip install -e .
```

If you want text embeddings:

```bash
pip install -e ".[embeddings]"
```

If you want the optional GCN:

```bash
pip install -e ".[gcn]"
```

Or everything:

```bash
pip install -e ".[all]"
```

## Data format (input JSON)

The pipeline consumes a **normalized JSON list**. Each record should look like:

```json
{
  "name": "Threat name",
  "created": "2023-08-01T12:34:56.123",
  "revision": 3,
  "adversary": "Some actor",
  "industries": ["Manufacturing"],
  "targeted_countries": ["IN", "GB"],
  "attack_ids": ["T1059.003"],
  "malware_families": ["AgentTesla"],
  "indicators": [
    {"indicator": "8.8.8.8", "type": "IPv4", "title": "", "description": "", "content": ""}
  ]
}
```

This matches what the original notebook expected from `ICS_IOCs_Updated.json`.

## Quickstart

### 1) Export from OTX (optional)

Put your API key in `.env`:

```bash
cp .env.example .env
# edit .env and set OTX_API_KEY=...
```

Export pulses:

```bash
python scripts/export_otx_to_json.py --out data/threats.json --limit 200
```

### 2) Run the pipeline

```bash
iocevaluator run --input data/threats.json --out out/
```

This will write:

- `out/attackers.csv` – attacker list (node order)
- `out/features.npy` – attacker feature matrix (N×D)
- `out/miis.npy` – attacker similarity/adjacency (N×N)
- `out/threats_normalized.json` – sanitised normalised copy

### 3) Run with GCN embeddings (optional)

```bash
iocevaluator run --input data/threats.json --out out/ --gcn --gcn-epochs 100
```

Outputs:

- `out/gcn_embeddings.npy` – node embeddings (N×D)
- `out/gcn_recon.npy` – reconstructed adjacency (N×N)
- `out/gcn_loss.csv` – training curve

## Final / manuscript-scale experiments (frozen-protocol preflight)

Every route that can launch a final or manuscript-scale run goes through `final_experiment_preflight` (`iocevaluator/protocol.py`):

```bash
python scripts/run_experiment.py --task models --source real --tikg T.json --labels L.json --severity S.json --paths P.json \
    --feature-policy policy.json --out results/final_models        # FINAL: clean git tree required, aborts before training on any failure
python scripts/run_experiment.py ... --dry-run                     # full preflight (clean-tree check bypassed), prints the summary, never trains
python scripts/run_experiment.py ... --allow-dirty                 # NON-FINAL diagnostic run: marked diagnostic, never reportable
iocevaluator experiment ...                                        # the same launcher through the CLI
```

`--feature-policy` (JSON `{"exclude_fields": [], "strip_terms": [], "source": "..."}`; `"source": "none required: <reason>"` if nothing is
excluded) is required in every mode; `--dry-run` and `--allow-dirty` relax only the clean-tree requirement.  The preflight summary and
the post-run integrity verdict are written into every `manifest.json` (`preflight`, `integrity`; outside `content_sha256`).
After the run `search_budget_hit` is re-checked on every path row; one hit fails the experiment and no result tables are written.
The legacy `iocevaluator evaluate` harness, the path-budget study and `rank --reference-paths` refuse manuscript-scale runs;
`scripts/run_scalability.py` runs the preflight for sizes >= 3,000 nodes.  Sensitivity configurations (alpha, tau, path-budget levels) are
never accepted as the primary configuration.

## What was fixed vs the notebook

- Removed all `...` placeholder fragments that caused **syntax errors**.
- Removed hard-coded API keys; uses **environment variables** (`OTX_API_KEY`).
- Replaced duplicated / inconsistent functions with a single tested implementation:
  - relation matrices are built directly from `Event` objects
  - MIIS aggregation is vectorised and numerically safe
- Converted the flow into a clean **package + CLI** with clear outputs.

## Notes

- If your source feed uses different indicator type labels, extend `TYPE_TO_BUCKET` in `event_builder.py`.
- For large datasets, start with `--no-text-emb` to avoid embedding downloads.

## metA4API implementation (manuscript methodology)

New modules (the original attacker-centric `run` pipeline is unchanged):

| Module | Purpose |
|---|---|
| `tikg.py` | 7 entity types, 9 relations, T1-T11 (+X1) validated triplets, OTX adapter |
| `metagraphs.py` | chi_1..chi_20 per Fig. 3 (base meta-paths, Hadamard / wrapped meta-graphs, Algorithm 1), uniform w_k = 1/20 |
| `megits.py` | weighted MeGiTS, leakage-safe per-fold adjacency |
| `models.py` | 2-layer GCN (sigmoid/BCE, dropout, wd, early stopping), depth 1-4, GAT, HGT-style |
| `prioritisation.py`, `attack_paths.py` | eigenvector centrality, CVSS fusion, path tracing/ranking, explanations |
| `splits.py`, `metrics.py`, `experiments.py` | campaign-isolated 10-fold x 5 seeds, all metrics, ablations, outputs |

```bash
iocevaluator build-tikg --input data/threats.json --extra extra.json --out tikg.json
iocevaluator evaluate --tikg tikg.json --labels labels.json --out results/ --variants ablation depth baselines
iocevaluator rank --tikg tikg.json --labels labels.json --out rank/ --oof-probs results/oof_probs_metA4API.npy \
    --severity severity.json --reference-paths reference_paths.json
python -m pytest
```

Real data (labels, devices/platforms, CVSS, reference paths) is **not included**: see `docs/DATA_REQUIREMENTS.md`.
Open manuscript/implementation mismatches: `docs/MANUSCRIPT_MAPPING.md`.

## License

MIT
