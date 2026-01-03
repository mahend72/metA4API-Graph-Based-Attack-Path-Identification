# metA4API-Graph-Based-Attack-Path-Identification

A small, reproducible Python repo that turns threat-feed IOC data (e.g., AlienVault OTX pulses) into:

- **Event graph** (attacker ↔ IOC types)
- **Meta-path commuting matrices**
- **MIIS/MPGIIS-style attacker similarity matrix** (Dice on meta-path counts)
- Optional **GCN autoencoder embeddings** (unsupervised)

This repo is a cleaned, modularised implementation based on the attached `IOCEvaluator.ipynb` notebook.

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
│  └─ export_otx_to_json.py       # helper: export OTX to normalized JSON
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
- `out/miis.npy` – attacker similarity / adjacency (N×N)
- `out/threats_normalized.json` – sanitized normalized copy

### 3) Run with GCN embeddings (optional)

```bash
iocevaluator run --input data/threats.json --out out/ --gcn --gcn-epochs 100
```

Outputs:

- `out/gcn_embeddings.npy` – node embeddings (N×D)
- `out/gcn_recon.npy` – reconstructed adjacency (N×N)
- `out/gcn_loss.csv` – training curve

## What was fixed vs the notebook

- Removed all `...` placeholder fragments that caused **syntax errors**.
- Removed hard-coded API keys; uses **environment variables** (`OTX_API_KEY`).
- Replaced duplicated / inconsistent functions with a single tested implementation:
  - relation matrices are built directly from `Event` objects
  - MIIS aggregation is vectorized and numerically safe
- Converted the notebook flow into a clean **package + CLI** with clear outputs.

## Notes

- If your source feed uses different indicator type labels, extend `TYPE_TO_BUCKET` in `event_builder.py`.
- For large datasets, start with `--no-text-emb` to avoid embedding downloads.

## License

MIT
