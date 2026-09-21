# Selective Gated Residual Neural Fusion (SGRPN) for Milling Surface Roughness

Reference implementation and analysis code for the manuscript:

> **Selective Gated Residual Neural Fusion with Group Split-Conformal Uncertainty Quantification for Milling Surface Roughness**

This repository contains the exact code that produced the frozen numerical results reported in the manuscript, together with the frozen analysis protocols used to generate them.

**Repository:** <https://github.com/zhushengluo/sgrpn-milling-roughness>

---

## What this repository contains

| Path | Contents |
|---|---|
| `src/roughness/sgrpn/` | The proposed method: selective gated residual fusion, group-isolated cross-fitting, group split-conformal interval calibration, scale ablation, and the post-audit corrective reanalysis |
| `src/roughness/scheme1/`, `src/roughness/scheme1_physics/` | Earlier baseline routes retained for comparison and ablation provenance |
| `configs/` | Frozen YAML configurations for every reported run (Phase A, Phase B, and the corrective reanalysis) |
| `tests/` | Unit and pipeline tests, including the tamper/consistency guards used to protect the frozen results |
| `scripts/` | Analysis, preprocessing, plotting, and document-generation scripts |
| `docs/paper/2026-09-10-sgrpn-split-conformal-corrective-protocol.md` | **The frozen corrective protocol with its input SHA-256 fingerprints** |
| `docs/superpowers/` | Design and implementation records, including the Phase A v1 invalidation audit |

---

## Analysis-protocol provenance

The analysis protocol was **frozen before the corresponding results were produced**, and the freeze points are externally verifiable in this repository's commit history:

| Freeze point | When | How to verify |
|---|---|---|
| Phase A protocol locked | **2026-08-21** (before the formal Phase A run on 2026-08-22) | `git log --oneline --grep="lock SGRPN phase A protocol"` → commit `5e52cd5` |
| Corrective protocol frozen | **2026-09-10T08:05:27.798Z** | `docs/paper/2026-09-10-sgrpn-split-conformal-corrective-protocol.md` (records the freeze timestamp and the SHA-256 fingerprints of all frozen inputs) |

**Scope of the claim, stated exactly as in the protocol:** the corrective analysis is a *post-audit corrective reanalysis of the existing dataset*. It is **not** a new physical experiment, **not** an independent replication, and **not** a publicly preregistered study. The manuscript restricts its claims accordingly.

---

## Requirements

- Python `>=3.12,<3.13`
- PyTorch with CUDA 12.8 for the GPU training paths (see `requirements/torch-cu128.txt`)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements/base.txt
pip install -r requirements/torch-cu128.txt   # optional: GPU training
pip install -e .
```

## Running the code

```bash
# unit and pipeline tests
python -m pytest tests -q

# entry points declared in pyproject.toml
roughness-sgrpn --help
roughness-scheme1 --help
roughness-scheme1-physics --help
```

The YAML files under `configs/` are the frozen definitions of each reported run. The code refuses to resume from metadata that does not match the frozen protocol fingerprint, so a configuration cannot be silently altered after the fact.

---

## Data availability

The raw machining and vibration measurement data are **not publicly released**. The code in this repository is provided so that the exact processing, cross-fitting, calibration, and evaluation pipeline can be inspected and reproduced on equivalent data.

Because the raw signals are not distributed, the pipeline cannot be re-executed end-to-end from this repository alone. Processed tables and figures are not redistributed here either; they are reported in the manuscript and its supplementary material.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

## Citation

See [`CITATION.cff`](CITATION.cff).
