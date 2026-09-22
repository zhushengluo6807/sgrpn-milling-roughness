# Selective Gated Residual Neural Fusion (SGRPN) for Milling Surface Roughness

Reference implementation and analysis code for the manuscript:

> **Selective Gated Residual Neural Fusion with Group Split-Conformal Uncertainty Quantification for Milling Surface Roughness**

This repository contains the exact code that produced the frozen numerical results reported in the manuscript, together with the frozen analysis protocols used to generate them.

**Repository:** <https://github.com/zhushengluo6807/sgrpn-milling-roughness>

---

## What this repository contains

| Path | Contents |
|---|---|
| `src/roughness/sgrpn/` | The proposed method: selective gated residual fusion, group-isolated cross-fitting, group split-conformal interval calibration, scale ablation, and the post-audit corrective reanalysis |
| `src/roughness/scheme1/`, `src/roughness/scheme1_physics/` | Earlier baseline routes retained for comparison and ablation provenance |
| `configs/` | Frozen YAML configurations for every reported run (Phase A, Phase B, and the corrective reanalysis) |
| `data/group_folds.csv` | **The frozen group partition** used by every reported run — see below |
| `tests/` | Unit and pipeline tests, including the tamper/consistency guards used to protect the frozen results |
| `scripts/` | Analysis, preprocessing, plotting, and document-generation scripts |
| `docs/paper/2026-09-10-sgrpn-split-conformal-corrective-protocol.md` | **The frozen corrective protocol with its input SHA-256 fingerprints** |
| `docs/superpowers/` | Design and implementation records, including the Phase A v1 invalidation audit |

---

## The frozen group partition

`data/group_folds.csv` is the group partition that every reported run consumed. It has three
columns — `sample_id, group_id, fold` — covering 586 samples in 212 anonymous groups, and it is
the file whose SHA-256 the Phase A run manifest registers as an immutable input.

```
sample_id,group_id,fold
v3_100_seg1,v3_100,1
v3_100_seg2,v3_100,1
...
```

- SHA-256 of the file bytes: `b63563da3adbfb8eac4694e22e618b309dd9d6f093377768bbef103eceec8a69`
- 586 samples, 212 groups, 5 folds (42–43 groups per fold; 113–121 samples per fold)
- **No group appears in more than one fold** — this is the group isolation the method relies on, and it can be checked directly from this file
- It contains **no roughness values, no process parameters, and no signal paths**; it is only the anonymous mapping from a sample to its group and to the fold that group was assigned to

Because the group assignment is fixed and published, the group-isolated cross-fitting and the
proper-training / calibration splits of the corrective analysis are reproducible from this
repository without access to the raw measurements. The file is stored byte for byte (see
`.gitattributes`), so the hash above can be reproduced on any platform.

---

## Analysis-protocol provenance

The analysis protocol was **frozen before the corresponding results were produced**, and the freeze points are externally verifiable in this repository's commit history:

| Freeze point | When | How to verify |
|---|---|---|
| Phase A protocol locked | **2026-08-21 17:04:51 +0800** — before the Phase A pipeline was implemented, and before the authoritative Phase A (v2) execution on 2026-08-23 | `git log --oneline --grep="lock SGRPN phase A protocol"` → commit `5e52cd5` |
| Corrective protocol frozen | **2026-09-10T08:05:27.798Z** | `docs/paper/2026-09-10-sgrpn-split-conformal-corrective-protocol.md` (records the freeze timestamp and the SHA-256 fingerprints of all frozen inputs) |

The frozen configurations are chained by the SHA-256 of their file bytes, so the chain can be
walked using the public files alone: `configs/sgrpn_phase_b.yaml` records the SHA-256 of
`configs/sgrpn_phase_a.yaml`; the corrective configuration records that of
`configs/sgrpn_phase_b.yaml`, which is in turn listed among the immutable inputs of the frozen
corrective protocol.

> **One Phase A execution was discarded.** An earlier run (`2026-08-22 22:41` →
> `2026-08-23 01:47`) was invalidated in full because its early-stopping selector and its
> inference predictor disagreed; the audit is recorded in
> `docs/superpowers/reports/2026-08-23-phase-a-v1-invalidation-and-v2-rerun.md` and that run
> **must not be cited as evidence**. Every Phase A result reported in the manuscript comes from
> the authoritative re-run of 2026-08-23.

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

The frozen group partition (`data/group_folds.csv`) and the frozen configurations (`configs/`) are released, so the group-isolated evaluation design itself is externally checkable; the raw signals are not.

Because the raw signals are not distributed, the pipeline cannot be re-executed end-to-end from this repository alone. Processed tables and figures are not redistributed here either; they are reported in the manuscript and its supplementary material.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

## Citation

See [`CITATION.cff`](CITATION.cff).
