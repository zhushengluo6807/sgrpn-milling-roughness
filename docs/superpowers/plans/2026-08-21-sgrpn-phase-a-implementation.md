# SGRPN Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and formally evaluate the leakage-safe Phase A SGRPN mean-prediction pipeline, then emit a machine-readable decision on whether Phase B is allowed.

**Architecture:** Add an isolated `roughness.sgrpn` package that reuses the existing manifest, fixed group folds, window index, signal loader, sample weights, and read-only M0 predictions. Precompute order-spectrum bags, train a Process-MLP with inner group cross-fitting, train vibration residual and comparison models, freeze the experts while fitting the trust gate, and evaluate all six models from outer-fold OOF predictions.

**Tech Stack:** Python 3.12.13, PyTorch 2.11.0, NumPy 2.5.1, pandas 2.3.3, scikit-learn 1.9.0, PyYAML, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-sgrpn-design.md`

## Global Constraints

- Run Python with `python`; do not mutate `python` or install global packages.
- Use the existing five outer folds from `outputs/first_round_baseline/folds.csv`; `group_id` must never cross train, validation, test, or calibration boundaries.
- Phase A uses only seed `20260723`, four inner group folds, at most 200 epochs, patience 20, and `1/split_count` sample weights.
- The order grid is exactly 0–90 order inclusive at 0.25 spacing, hence 361 bins; input windows are 25,600 samples at 25.6 kHz.
- Recompute duration as `row_count / 25600`; retain source metadata only for audit.
- Exclude `version` from every model input; use it only in diagnostic breakdowns.
- Train-time Ch9/Ch10 swapping uses probability 0.5; inference averages original and swapped predictions.
- Existing `outputs/scheme1/` and `outputs/scheme1_physics/` are read-only; write only under `outputs/sgrpn/phase_a/`.
- Do not implement Phase B in this plan. Write its separate plan only if `outputs/sgrpn/phase_a/evaluation/acceptance.json` says `proceed_to_phase_b: true`.
- The current `.git` directory is empty and is not a valid repository. Do not initialize Git automatically; commit steps are conditional on the user supplying a valid repository before execution.

## File Map

| Path | Responsibility |
|---|---|
| `src/roughness/sgrpn/config.py` | Load and validate the fixed Phase A protocol |
| `src/roughness/sgrpn/data.py` | Load manifest/folds/windows, audit duration, build process features |
| `src/roughness/sgrpn/order_spectrum.py` | Compute, cache, scale, and validate order-spectrum bags and quality features |
| `src/roughness/sgrpn/dataset.py` | Expose variable-window bags and deterministic swap augmentation |
| `src/roughness/sgrpn/models.py` | P1, V1, F1, R1, G1 modules and outputs |
| `src/roughness/sgrpn/crossfit.py` | Inner group splits and P1 OOF residual generation |
| `src/roughness/sgrpn/training.py` | Epoch selection, refit, fold orchestration, checkpoints, fingerprints |
| `src/roughness/sgrpn/evaluation.py` | Metrics, group bootstrap, negative transfer, acceptance gate |
| `src/roughness/sgrpn/reporting.py` | CSV/JSON tables and publication diagnostic figures |
| `src/roughness/sgrpn/cli.py` | `audit`, `features`, `train-phase-a`, `evaluate-phase-a`, `run-phase-a` |
| `configs/sgrpn_phase_a.yaml` | Fixed paths and hyperparameters |
| `tests/sgrpn/` | Unit, leakage, integration, and immutability tests |

---

### Task 1: Lock the Phase A configuration and CLI registration

**Files:**
- Create: `src/roughness/sgrpn/__init__.py`
- Create: `src/roughness/sgrpn/config.py`
- Create: `configs/sgrpn_phase_a.yaml`
- Create: `tests/sgrpn/__init__.py`
- Create: `tests/sgrpn/test_config.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: existing project-relative data and output paths.
- Produces: `SGRPNConfig`, `load_sgrpn_config(path)`, `config_fingerprint(config, phase)` and console script `roughness-sgrpn`.

- [ ] **Step 1: Write failing configuration tests**

```python
from pathlib import Path
import pytest

from roughness.sgrpn.config import load_sgrpn_config


def test_phase_a_protocol_is_locked():
    cfg = load_sgrpn_config(Path("configs/sgrpn_phase_a.yaml"))
    assert cfg.sample_rate_hz == 25600
    assert cfg.window_samples == 25600
    assert cfg.order_min == 0.0
    assert cfg.order_max == 90.0
    assert cfg.order_step == 0.25
    assert cfg.seeds == (20260723,)
    assert cfg.inner_splits == 4
    assert cfg.bootstrap_repetitions == 10000


def test_rejects_phase_a_with_multiple_seeds(tmp_path):
    text = Path("configs/sgrpn_phase_a.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(text.replace("seeds: [20260723]", "seeds: [1, 2]"), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly seed 20260723"):
        load_sgrpn_config(path)
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

Run:

```powershell
python -m pytest tests/sgrpn/test_config.py -v
```

Expected: collection fails because `roughness.sgrpn.config` does not exist.

- [ ] **Step 3: Add the frozen dataclass and loader**

Implement these exact public names:

```python
@dataclass(frozen=True)
class SGRPNConfig:
    manifest_path: Path
    folds_path: Path
    window_index_path: Path
    m0_oof_path: Path
    output_dir: Path
    sample_rate_hz: int
    window_samples: int
    order_min: float
    order_max: float
    order_step: float
    seeds: Sequence[int]
    inner_splits: int
    max_epochs: int
    patience: int
    process_learning_rate: float
    signal_learning_rate: float
    gate_learning_rate: float
    weight_decay: float
    huber_delta_um: float
    gate_penalty: float
    correction_penalty: float
    bootstrap_repetitions: int


# Public signatures
# load_sgrpn_config(path: str | Path) -> SGRPNConfig
# config_fingerprint(config: SGRPNConfig, phase: str) -> str
```

The loader must resolve paths relative to the YAML directory, verify all four read-only input files exist, require the exact Phase A seed tuple, and require `(order_max-order_min)/order_step + 1 == 361`.

- [ ] **Step 4: Write the fixed YAML**

```yaml
manifest_path: ../outputs/first_round_baseline/manifest.csv
folds_path: ../outputs/first_round_baseline/folds.csv
window_index_path: ../outputs/scheme1/window_index.csv
m0_oof_path: ../outputs/scheme1/classic/oof_predictions.csv
output_dir: ../outputs/sgrpn/phase_a
sample_rate_hz: 25600
window_samples: 25600
order_min: 0.0
order_max: 90.0
order_step: 0.25
seeds: [20260723]
inner_splits: 4
max_epochs: 200
patience: 20
process_learning_rate: 0.001
signal_learning_rate: 0.0003
gate_learning_rate: 0.001
weight_decay: 0.0001
huber_delta_um: 0.10
gate_penalty: 0.001
correction_penalty: 0.01
bootstrap_repetitions: 10000
```

- [ ] **Step 5: Register the command without implementing handlers**

Add to `[project.scripts]`:

```toml
roughness-sgrpn = "roughness.sgrpn.cli:main"
```

Create `cli.py` only in Task 8; until then the config unit tests must not invoke the entry point.

- [ ] **Step 6: Run the configuration tests**

Run the command from Step 2. Expected: both tests pass.

- [ ] **Step 7: Commit when Git is available**

First run `git rev-parse --is-inside-work-tree`. If it returns `true`, run:

```powershell
git add pyproject.toml configs/sgrpn_phase_a.yaml src/roughness/sgrpn/__init__.py src/roughness/sgrpn/config.py tests/sgrpn/__init__.py tests/sgrpn/test_config.py
git commit -m "feat: lock SGRPN phase A protocol"
```

If it still reports “not a git repository,” do not initialize Git; preserve the passing test output as the task checkpoint.

---

### Task 2: Build leakage-safe data contracts and duration audit

**Files:**
- Create: `src/roughness/sgrpn/data.py`
- Create: `tests/sgrpn/test_data.py`

**Interfaces:**
- Consumes: `SGRPNConfig`; existing `roughness.scheme1.folds.validate_outer_folds`.
- Produces: `DataBundle`, `load_data_bundle(config)`, `build_process_features(frame)`, `outer_indices(bundle, fold)`, and `write_data_audit(bundle, path)`.

- [ ] **Step 1: Write failing tests for process columns, duration, and group isolation**

```python
import numpy as np
import pandas as pd
import pytest

from roughness.sgrpn.data import build_process_features, recompute_duration, validate_group_split


def test_process_features_are_exactly_nine_columns():
    frame = pd.DataFrame({"n_rpm": [4000.0], "fz_mm_per_tooth": [0.05], "ap_mm": [1.0]})
    features = build_process_features(frame)
    np.testing.assert_allclose(
        features[0],
        [4000.0, 0.05, 1.0, 16_000_000.0, 0.0025, 1.0, 200.0, 4000.0, 0.05],
    )
    assert features.shape == (1, 9)


def test_duration_uses_row_count_not_metadata():
    assert recompute_duration(row_count=25600, sample_rate_hz=25600) == 1.0


def test_group_overlap_fails_fast():
    with pytest.raises(ValueError, match="group leakage"):
        validate_group_split(np.array(["g1", "g2"]), np.array(["g2", "g3"]))
```

- [ ] **Step 2: Run the targeted tests and verify failure**

```powershell
python -m pytest tests/sgrpn/test_data.py -v
```

Expected: import fails because `roughness.sgrpn.data` is absent.

- [ ] **Step 3: Implement immutable data loading**

Use these interfaces:

```python
@dataclass(frozen=True)
class DataBundle:
    manifest: pd.DataFrame
    folds: pd.DataFrame
    windows: pd.DataFrame
    fold_audit: dict[str, int]
    duration_audit: pd.DataFrame


# Public signatures
# recompute_duration(row_count: int, sample_rate_hz: int) -> float
# build_process_features(frame: pd.DataFrame) -> np.ndarray
# validate_group_split(train_groups: np.ndarray, test_groups: np.ndarray) -> None
# load_data_bundle(config: SGRPNConfig) -> DataBundle
# outer_indices(bundle: DataBundle, fold: int) -> tuple[np.ndarray, np.ndarray]
# write_data_audit(bundle: DataBundle, path: str | Path) -> Path
```

`load_data_bundle` must require `sample_id`, `group_id`, `signal_path`, `n_rpm`, `fz_mm_per_tooth`, `ap_mm`, `ra_1`, `ra_2`, `ra_3`, `ra_mean`, `sample_weight`, `split_count`, `version`, and `duration_s`. For every signal CSV, count data rows through the existing `load_signal_csv`; record `duration_source_s`, `duration_recomputed_s`, `absolute_difference_s`, and `mismatch_over_1ms`. Do not modify the manifest file.

- [ ] **Step 4: Add tests against copied miniature manifests**

Create two three-channel CSV fixtures with 25,600 rows, a two-row manifest, a fold file, and matching window rows under `tmp_path`. Assert that loading rejects a missing Ra repeat, rejects a non-finite process value, records a duration mismatch, and leaves the source manifest bytes unchanged.

- [ ] **Step 5: Run Task 2 tests and the reused fold tests**

```powershell
python -m pytest tests/sgrpn/test_data.py tests/scheme1/test_folds.py -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit when Git is available**

```powershell
git add src/roughness/sgrpn/data.py tests/sgrpn/test_data.py
git commit -m "feat: add SGRPN data audit contracts"
```

Skip only when `git rev-parse --is-inside-work-tree` is not `true`.

---

### Task 3: Precompute order-spectrum bags and signal-quality features

**Files:**
- Create: `src/roughness/sgrpn/order_spectrum.py`
- Create: `tests/sgrpn/test_order_spectrum.py`

**Interfaces:**
- Consumes: `DataBundle`, `SGRPNConfig`, and `roughness.scheme1.signals.load_signal_csv`.
- Produces: `OrderSpectrumCache`, `SpectrumScaler`, `QualityScaler`, `order_grid`, `window_to_order_spectrum`, `signal_quality_features`, `build_order_cache`, `save_order_cache`, `load_order_cache`, `fit_spectrum_scaler`, and `fit_quality_scaler`.

- [ ] **Step 1: Write the known-order failing test**

```python
import numpy as np

from roughness.sgrpn.order_spectrum import order_grid, window_to_order_spectrum


def test_sixth_order_sine_peaks_near_six():
    fs = 25600
    n_rpm = 4000.0
    shaft_hz = n_rpm / 60.0
    t = np.arange(fs) / fs
    one = np.sin(2 * np.pi * 6.0 * shaft_hz * t)
    signal = np.column_stack([one, 0.5 * one, 0.25 * one])
    spectrum = window_to_order_spectrum(signal, n_rpm, fs, 0.0, 90.0, 0.25)
    grid = order_grid(0.0, 90.0, 0.25)
    assert spectrum.shape == (3, 361)
    assert abs(grid[np.argmax(spectrum[0])] - 6.0) <= 0.25
```

- [ ] **Step 2: Write cache and horizontal-symmetry tests**

```python
def test_quality_vector_is_horizontal_swap_invariant():
    signal = np.column_stack([
        np.linspace(-1.0, 1.0, 25600),
        np.linspace(1.0, -1.0, 25600),
        np.sin(np.linspace(0.0, 20.0, 25600)),
    ])
    q_ab = signal_quality_features(signal)
    q_ba = signal_quality_features(signal[:, [1, 0, 2]])
    np.testing.assert_allclose(q_ab, q_ba)
    assert q_ab.shape == (7,)


def test_scaler_uses_only_requested_segments():
    cache = OrderSpectrumCache(
        segment_ids=("s1", "s2"),
        spectra=np.stack([np.ones((3, 361)), np.full((3, 361), 100.0)]).astype(np.float32),
        offsets=np.array([0, 1, 2], dtype=np.int64),
        quality=np.zeros((2, 7), dtype=np.float32),
        durations_s=np.ones(2),
    )
    scaler = fit_spectrum_scaler(cache, ["s1"])
    expected = cache.bag("s1").mean(axis=0)
    np.testing.assert_allclose(scaler.mean, expected)
```

- [ ] **Step 3: Run the tests and confirm missing imports**

```powershell
python -m pytest tests/sgrpn/test_order_spectrum.py -v
```

- [ ] **Step 4: Implement the exact cache schema**

```python
@dataclass(frozen=True)
class OrderSpectrumCache:
    segment_ids: Sequence[str]
    spectra: np.ndarray       # [all_windows, 3, 361], float32
    offsets: np.ndarray       # [n_segments + 1], int64
    quality: np.ndarray       # [n_segments, 7], float32
    durations_s: np.ndarray   # [n_segments], float64

    # bag(segment_id: str) -> np.ndarray returns spectra[offsets[i]:offsets[i+1]]
    # quality_row(segment_id: str) -> np.ndarray returns quality[i]


@dataclass(frozen=True)
class SpectrumScaler:
    mean: np.ndarray          # [3, 361]
    scale: np.ndarray         # [3, 361]


@dataclass(frozen=True)
class QualityScaler:
    mean: np.ndarray          # [7]
    scale: np.ndarray         # [7]
```

`window_to_order_spectrum` must validate `[25600, 3]`, subtract each channel mean, apply `np.hanning(25600)`, use `np.fft.rfft`, compute power, convert frequency to order with `n_rpm/60`, interpolate `np.log1p(power)`, and reject non-positive rpm. Use the existing window index rather than inventing new partial windows.

- [ ] **Step 5: Implement cache serialization and fingerprinting**

Write `features/order_spectrum_cache.npz` and `features/order_spectrum_cache.json` under the new output directory. JSON must record grid values, sample rate, manifest SHA-256, window-index SHA-256, segment count, window count, mismatch count, and cache SHA-256. Loading must reject an incompatible grid or fingerprint.

- [ ] **Step 6: Run the order-spectrum tests**

Run the command from Step 3. Expected: all tests pass and no test reads the formal dataset.

- [ ] **Step 7: Commit when Git is available**

```powershell
git add src/roughness/sgrpn/order_spectrum.py tests/sgrpn/test_order_spectrum.py
git commit -m "feat: add fixed-grid order spectrum cache"
```

---

### Task 4: Add variable-window datasets and swap-ensemble inference

**Files:**
- Create: `src/roughness/sgrpn/dataset.py`
- Create: `tests/sgrpn/test_dataset.py`

**Interfaces:**
- Consumes: `OrderSpectrumCache`, `SpectrumScaler`, `QualityScaler`, nine-dimensional process arrays, and caller-supplied targets.
- Produces: `OrderBagDataset`, `collate_order_bags`, and `swap_horizontal`.

- [ ] **Step 1: Write failing padding and weighting tests**

```python
def item_with_windows(count: int, weight: float) -> dict:
    return {
        "spectrum": torch.zeros(count, 3, 361),
        "process": torch.zeros(9),
        "quality": torch.zeros(7),
        "target": torch.tensor(0.5),
        "sample_weight": torch.tensor(weight),
        "sample_id": f"s{count}",
        "group_id": f"g{count}",
    }


def test_collate_preserves_weights_and_masks_variable_bags():
    batch = collate_order_bags([item_with_windows(1, weight=1.0), item_with_windows(3, weight=0.5)])
    assert batch["spectrum"].shape == (2, 3, 3, 361)
    assert batch["window_mask"].tolist() == [[True, False, False], [True, True, True]]
    assert batch["sample_weight"].tolist() == [1.0, 0.5]


def test_horizontal_swap_moves_only_first_two_channels():
    batch = collate_order_bags([item_with_windows(2, weight=1.0)])
    batch["spectrum"][:, :, 0] = 1.0
    batch["spectrum"][:, :, 1] = 2.0
    batch["spectrum"][:, :, 2] = 3.0
    swapped = swap_horizontal(batch)
    torch.testing.assert_close(swapped["spectrum"][:, :, 0], batch["spectrum"][:, :, 1])
    torch.testing.assert_close(swapped["spectrum"][:, :, 1], batch["spectrum"][:, :, 0])
    torch.testing.assert_close(swapped["spectrum"][:, :, 2], batch["spectrum"][:, :, 2])
```

- [ ] **Step 2: Run the targeted tests and verify failure**

```powershell
python -m pytest tests/sgrpn/test_dataset.py -v
```

- [ ] **Step 3: Implement the dataset contract**

```python
class OrderBagDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        cache: OrderSpectrumCache,
        spectrum_scaler: SpectrumScaler,
        quality_scaler: QualityScaler,
        process: np.ndarray,
        targets: np.ndarray,
        augment_horizontal_swap: bool,
    ) -> None:
        self.augment_horizontal_swap = bool(augment_horizontal_swap)


# Public signatures
# collate_order_bags(items: list[dict]) -> dict
# swap_horizontal(batch: dict) -> dict
```

Each item must contain `spectrum`, `process`, `quality`, `target`, `sample_weight`, `sample_id`, and `group_id`. Validate finite arrays and reject a bag with zero windows. Return copies from `swap_horizontal`; never mutate a caller-owned batch.

- [ ] **Step 4: Add deterministic augmentation tests**

Patch `torch.rand` to return 0.25 and 0.75 in separate tests. Assert swapping occurs only below 0.5 and only when `augment_horizontal_swap=True`.

- [ ] **Step 5: Run dataset tests**

```powershell
python -m pytest tests/sgrpn/test_dataset.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit when Git is available**

```powershell
git add src/roughness/sgrpn/dataset.py tests/sgrpn/test_dataset.py
git commit -m "feat: add SGRPN order bag dataset"
```

---

### Task 5: Implement P1, V1, F1, R1, G1 and the fixed losses

**Files:**
- Create: `src/roughness/sgrpn/models.py`
- Create: `tests/sgrpn/test_models.py`

**Interfaces:**
- Consumes: padded order bags `[B,W,3,361]`, masks `[B,W]`, process `[B,9]`, quality `[B,7]`.
- Produces: `ModelOutput`, `ProcessMLP`, `OrderSpectrumEncoder`, `VibrationOnlyModel`, `DirectFusionModel`, `ResidualExpert`, `ResidualFusionModel`, `SelectiveGatedModel`, `average_swap_predictions`, `weighted_huber`, and `sgrpn_loss`.

- [ ] **Step 1: Write architecture and boundary tests**

```python
def test_process_mlp_returns_one_mean_per_row():
    model = ProcessMLP()
    assert model(torch.zeros(4, 9)).shape == (4,)


def test_encoder_returns_64_dimensional_segment_embedding():
    encoder = OrderSpectrumEncoder()
    out = encoder(torch.zeros(2, 3, 3, 361), torch.ones(2, 3, dtype=torch.bool))
    assert out.shape == (2, 64)


def test_gated_model_has_exact_fallback_boundaries():
    base = torch.tensor([1.0, 2.0])
    delta = torch.tensor([0.2, -0.3])
    torch.testing.assert_close(combine_prediction(base, delta, torch.zeros(2)), base)
    torch.testing.assert_close(combine_prediction(base, delta, torch.ones(2)), base + delta)
```

- [ ] **Step 2: Write freeze and loss tests**

```python
def test_gate_training_freezes_both_experts():
    model = SelectiveGatedModel(ProcessMLP(), ResidualExpert(OrderSpectrumEncoder()))
    model.freeze_experts()
    assert not any(p.requires_grad for p in model.process_expert.parameters())
    assert not any(p.requires_grad for p in model.residual_expert.parameters())
    assert any(p.requires_grad for p in model.gate.parameters())


def test_sgrpn_loss_matches_registered_formula():
    output = ModelOutput(
        prediction=torch.tensor([1.2]),
        process_mean=torch.tensor([1.0]),
        residual=torch.tensor([0.4]),
        gate=torch.tensor([0.5]),
        embedding=torch.zeros(1, 64),
    )
    loss = sgrpn_loss(output, torch.tensor([1.0]), torch.tensor([1.0]), delta=0.1)
    expected = weighted_huber(output.prediction, torch.tensor([1.0]), torch.tensor([1.0]), 0.1)
    expected = expected + 1e-3 * 0.5**2 + 1e-2 * (0.5 * 0.4) ** 2
    torch.testing.assert_close(loss, expected)
```

- [ ] **Step 3: Run tests and confirm missing implementation**

```powershell
python -m pytest tests/sgrpn/test_models.py -v
```

- [ ] **Step 4: Implement focused model units**

Use this output contract throughout all later tasks:

```python
@dataclass
class ModelOutput:
    prediction: torch.Tensor
    process_mean: torch.Tensor | None = None
    residual: torch.Tensor | None = None
    gate: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
```

Implement the exact Process-MLP and three-layer CNN from the spec. Use BatchNorm and ReLU after each convolution, adaptive average pooling to 64 dimensions, masked mean across windows, and a `64→32→1` residual head. Implement:

```python
def combine_prediction(base: Tensor, delta: Tensor, gate: Tensor) -> Tensor:
    return base + gate * delta


# weighted_huber(prediction: Tensor, target: Tensor, weight: Tensor, delta: float) -> Tensor


def sgrpn_loss(output: ModelOutput, target: Tensor, weight: Tensor, delta: float) -> Tensor:
    return (
        weighted_huber(output.prediction, target, weight, delta)
        + 1e-3 * output.gate.square().mean()
        + 1e-2 * (output.gate * output.residual).square().mean()
    )
```

The gate input must be exactly `torch.cat([process, embedding, quality], dim=1)` with width 80 and architecture `Linear(80,16)→ReLU→Linear(16,1)→Sigmoid`.

Implement `average_swap_predictions(model: nn.Module, batch: dict) -> ModelOutput` here, after `ModelOutput` exists. It must call the model once with the original batch and once with `dataset.swap_horizontal(batch)`, then average every non-`None` tensor field. It must not mutate the input batch.

- [ ] **Step 5: Test input validation and parameter freezing**

Add explicit tests for wrong process width, wrong quality width, zero-window masks, non-three-channel spectra, and parameter equality before/after one gate optimizer step.

- [ ] **Step 6: Run model tests**

```powershell
python -m pytest tests/sgrpn/test_models.py -v
```

Expected: all tests pass on CPU.

- [ ] **Step 7: Commit when Git is available**

```powershell
git add src/roughness/sgrpn/models.py tests/sgrpn/test_models.py
git commit -m "feat: add selective gated residual models"
```

---

### Task 6: Generate P1 cross-fitted residual targets

**Files:**
- Create: `src/roughness/sgrpn/crossfit.py`
- Create: `tests/sgrpn/test_crossfit.py`

**Interfaces:**
- Consumes: outer-train manifest rows, raw nine-dimensional process features, targets, weights, four inner splits, and a process trainer callback.
- Produces: `ProcessScaler`, `ProcessOOFResult`, `fit_process_scaler`, `generate_process_oof`, and `median_best_epoch`.

- [ ] **Step 1: Write failing split and assignment tests**

```python
def make_frame(groups: int, rows_per_group: int) -> pd.DataFrame:
    group_id = [f"g{g}" for g in range(groups) for _ in range(rows_per_group)]
    count = len(group_id)
    return pd.DataFrame({
        "group_id": group_id,
        "n_rpm": np.linspace(4000.0, 8000.0, count),
        "fz_mm_per_tooth": np.linspace(0.03, 0.12, count),
        "ap_mm": np.linspace(0.5, 2.0, count),
        "ra_mean": np.linspace(0.2, 1.2, count),
        "sample_weight": np.ones(count),
    })


def test_each_group_receives_one_prediction_from_unseen_groups():
    frame = make_frame(groups=8, rows_per_group=2)
    seen = []

    def trainer(x_train, y_train, w_train, x_valid, train_groups, valid_groups):
        assert set(train_groups).isdisjoint(valid_groups)
        seen.extend(valid_groups)
        return np.full(len(x_valid), y_train.mean()), 7

    result = generate_process_oof(frame, build_process_features(frame), trainer, n_splits=4, seed=20260723)
    assert np.isfinite(result.prediction).all()
    assert result.assignment_count.tolist() == [1] * len(frame)
    assert sorted(seen) == sorted(frame.group_id.tolist())


def test_scaler_fits_only_inner_train_rows():
    x = np.array([[0.0] * 9, [10.0] * 9, [1000.0] * 9])
    scaler = fit_process_scaler(x, np.array([0, 1]))
    np.testing.assert_allclose(scaler.mean, np.full(9, 5.0))
```

- [ ] **Step 2: Run tests and verify failure**

```powershell
python -m pytest tests/sgrpn/test_crossfit.py -v
```

- [ ] **Step 3: Implement exact cross-fit result types**

```python
@dataclass(frozen=True)
class ProcessScaler:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


@dataclass(frozen=True)
class ProcessOOFResult:
    prediction: np.ndarray
    residual: np.ndarray
    assignment_count: np.ndarray
    best_epochs: Sequence[int]
    inner_fold: np.ndarray


ProcessTrainer = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, int],
]
```

Use `roughness.scheme1.crossfit.make_group_inner_splits`. For each inner fold, fit `ProcessScaler` on the inner training indices only, pass scaled train/validation arrays to the callback, assign exactly one validation prediction per row, and store the inner fold number. Compute residual as `ra_mean - prediction`. Raise if an index is assigned zero or multiple times, groups overlap, or a prediction is non-finite.

- [ ] **Step 4: Implement the real Process-MLP callback**

Add `fit_process_inner_fold(x_train, y_train, w_train, x_valid, train_groups, valid_groups) -> tuple[np.ndarray, int]`. It must train `ProcessMLP` with weighted Huber, AdamW at `1e-3`, weight decay `1e-4`, maximum 200 epochs, patience 20, and restore the epoch with minimum inner-validation weighted Huber. `median_best_epoch` must round the median of four positive epochs to the nearest integer, with a minimum of one.

- [ ] **Step 5: Add a label-permutation leakage test**

Construct four groups with extreme labels. Change labels in one validation group only and assert predictions for the other validation groups do not change when using a deterministic trainer. This proves each held-out label is absent from its predicting fit.

- [ ] **Step 6: Run cross-fit tests**

```powershell
python -m pytest tests/sgrpn/test_crossfit.py tests/scheme1/test_crossfit.py -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit when Git is available**

```powershell
git add src/roughness/sgrpn/crossfit.py tests/sgrpn/test_crossfit.py
git commit -m "feat: add process MLP cross-fitting"
```

---

### Task 7: Orchestrate leakage-safe Phase A fold training

**Files:**
- Create: `src/roughness/sgrpn/training.py`
- Create: `tests/sgrpn/test_training.py`

**Interfaces:**
- Consumes: `SGRPNConfig`, `DataBundle`, `OrderSpectrumCache`, model constructors, and an outer fold ID.
- Produces: `RunFingerprint`, `EpochSelection`, `FoldArtifacts`, `select_epochs_group_cv`, `refit_component`, `run_phase_a_fold`, and `run_phase_a`.

- [ ] **Step 1: Write failing epoch-selection tests**

```python
def test_epoch_selection_uses_median_inner_best_epoch():
    result = EpochSelection(best_epochs=(3, 9, 5, 7))
    assert result.refit_epochs == 6


def test_resume_requires_exact_fingerprint(tmp_path):
    marker = tmp_path / "complete.json"
    marker.write_text('{"fingerprint":"old"}', encoding="utf-8")
    assert completed_fold_matches(marker, "new") is False
```

- [ ] **Step 2: Write the fold output schema test**

```python
EXPECTED = {
    "sample_id", "group_id", "version", "fold", "seed", "model",
    "target", "prediction", "sample_weight", "process_mean", "residual", "gate",
}


def test_fold_predictions_have_registered_schema():
    rows = []
    for model in ("P1", "V1", "F1", "R1", "G1"):
        rows.append({
            "sample_id": "s1", "group_id": "g1", "version": "v3", "fold": 0,
            "seed": 20260723, "model": model, "target": 0.5, "prediction": 0.5,
            "sample_weight": 1.0, "process_mean": 0.5, "residual": np.nan, "gate": np.nan,
        })
    predictions = pd.DataFrame(rows)
    assert set(predictions.columns) == EXPECTED
    assert set(predictions.model.unique()) == {"P1", "V1", "F1", "R1", "G1"}
```

- [ ] **Step 3: Run tests and verify missing implementation**

```powershell
python -m pytest tests/sgrpn/test_training.py -v
```

- [ ] **Step 4: Implement run metadata and atomic checkpoints**

```python
@dataclass(frozen=True)
class RunFingerprint:
    value: str
    config_sha256: str
    manifest_sha256: str
    folds_sha256: str
    cache_sha256: str


@dataclass(frozen=True)
class EpochSelection:
    best_epochs: Sequence[int]

    @property
    def refit_epochs(self) -> int:
        return max(1, round(float(np.median(self.best_epochs))))


@dataclass(frozen=True)
class FoldArtifacts:
    predictions: pd.DataFrame
    checkpoint_paths: dict[str, Path]
    history_paths: dict[str, Path]
    fingerprint: RunFingerprint
```

Write JSON and checkpoint files through a sibling temporary path followed by `Path.replace`; a partial file must never be treated as complete. A completed fold is reusable only if fold, seed, model set, and fingerprint all match.

- [ ] **Step 5: Implement common inner epoch selection**

`select_epochs_group_cv(component_factory, dataset_factory, inner_splits, train_step, validation_step)` must fit spectrum and quality scalers on each inner-train subset, train on that subset, validate on the disjoint subset, and return four best epochs. `refit_component` must fit new scalers on the complete outer-train subset and train a fresh component for the median epoch count.

- [ ] **Step 6: Implement the fixed fold sequence**

`run_phase_a_fold(config, bundle, cache, fold, seed, device)` must execute exactly:

1. Assert outer train/test group disjointness.
2. Generate P1 OOF predictions and residual targets inside outer train.
3. Refit P1 on all outer-train rows and predict outer test.
4. Select/refit V1 on `ra_mean`.
5. Select/refit F1 on `ra_mean` using process plus vibration.
6. Select/refit the ResidualExpert on the P1 OOF residual target; combine with frozen final P1 to produce R1.
7. Copy P1 and ResidualExpert into G1, freeze both, select/refit only the 80→16→1 gate using `sgrpn_loss`.
8. Predict outer test with original/swapped spectrum averaging for V1/F1/R1/G1.
9. Save one prediction row per outer-test segment and model.

Never let G1 update expert parameters. Before and after gate training, compare every expert state-dict tensor with `torch.equal`; raise on any difference.

- [ ] **Step 7: Add sequencing tests with injected tiny models**

Use eight groups and one spectrum window per sample. Inject trainers that record sample IDs instead of performing expensive optimization. Assert no outer-test ID reaches any fit callback, residual targets equal `target - p1_oof`, gate training begins only after experts are marked frozen, and every outer-test row appears once per model.

- [ ] **Step 8: Add CPU integration coverage**

Run one outer fold with `max_epochs=2`, `patience=1`, four tiny groups per inner split, and the real model classes. Assert all predictions are finite, gates lie in `[0,1]`, scalers are written under the new output path, and no file appears under either legacy output directory.

- [ ] **Step 9: Run training tests**

```powershell
python -m pytest tests/sgrpn/test_training.py -v
```

Expected: all tests pass on CPU; the integration fixture completes without reading formal signal files.

- [ ] **Step 10: Commit when Git is available**

```powershell
git add src/roughness/sgrpn/training.py tests/sgrpn/test_training.py
git commit -m "feat: orchestrate SGRPN phase A folds"
```

---

### Task 8: Evaluate acceptance, generate reports, and expose the CLI

**Files:**
- Create: `src/roughness/sgrpn/evaluation.py`
- Create: `src/roughness/sgrpn/reporting.py`
- Create: `src/roughness/sgrpn/cli.py`
- Create: `tests/sgrpn/test_evaluation.py`
- Create: `tests/sgrpn/test_reporting.py`
- Create: `tests/sgrpn/test_cli.py`

**Interfaces:**
- Consumes: complete P1/V1/F1/R1/G1 OOF predictions and read-only M0 OOF predictions.
- Produces: `weighted_metrics`, `negative_transfer`, `paired_group_bootstrap`, `AcceptanceInputs`, `build_acceptance_inputs`, `assess_phase_a`, `write_phase_a_report`, `build_parser`, and `main`.

`negative_transfer` returns `NegativeTransferResult(raw_rate: float, material_rate: float, group_count: int)`; the material margin is exactly 0.01 μm of group-mean absolute-error excess over P1. `paired_group_bootstrap` returns `BootstrapResult(point_estimate: float, lower: float, upper: float, repetitions: int, resampling_unit: str)`.

- [ ] **Step 1: Write failing metric and negative-transfer tests**

```python
def test_material_negative_transfer_uses_point_zero_one_um():
    paired = pd.DataFrame({
        "group_id": ["a", "b"],
        "p1_abs_error": [0.10, 0.10],
        "candidate_abs_error": [0.105, 0.121],
    })
    result = negative_transfer(paired, material_margin_um=0.01)
    assert result.raw_rate == 1.0
    assert result.material_rate == 0.5


def test_bootstrap_resamples_groups_not_segments():
    rows = []
    for group, target in (("g1", 0.5), ("g2", 1.0), ("g3", 1.5)):
        rows.extend([
            {"group_id": group, "model": "P1", "target": target, "prediction": target + 0.1, "sample_weight": 1.0},
            {"group_id": group, "model": "G1", "target": target, "prediction": target + 0.05, "sample_weight": 1.0},
        ])
    predictions = pd.DataFrame(rows)
    result = paired_group_bootstrap(predictions, baseline="P1", candidate="G1", repetitions=50, seed=20260723)
    assert result.resampling_unit == "group_id"
    assert result.repetitions == 50
```

- [ ] **Step 2: Write the exact acceptance-gate tests**

```python
def test_phase_b_proceeds_on_mean_gain_path():
    decision = assess_phase_a(AcceptanceInputs(
        p1_mae_ratio_to_m0=1.04, p1_r2_drop_from_m0=0.01,
        g1_mae_ratio_to_p1=0.98, g1_rmse_ratio_to_p1=0.99, g1_r2_drop_from_p1=-0.01,
        g1_fold_wins=3, transfer_reduction_vs_f1=0.0, transfer_reduction_vs_r1=0.0,
        gate_median=0.3, gate_p95=0.8,
    ))
    assert decision.proceed_to_phase_b is True
    assert decision.path == "mean_improvement"


def test_phase_b_stops_when_gate_collapses_without_benefit():
    decision = assess_phase_a(AcceptanceInputs(
        p1_mae_ratio_to_m0=1.00, p1_r2_drop_from_m0=0.0,
        g1_mae_ratio_to_p1=1.00, g1_rmse_ratio_to_p1=1.00, g1_r2_drop_from_p1=0.0,
        g1_fold_wins=2, transfer_reduction_vs_f1=0.02, transfer_reduction_vs_r1=0.02,
        gate_median=0.01, gate_p95=0.10,
    ))
    assert decision.proceed_to_phase_b is False
    assert "collapsed" in decision.reasons
```

- [ ] **Step 3: Run evaluation tests and verify failure**

```powershell
python -m pytest tests/sgrpn/test_evaluation.py -v
```

- [ ] **Step 4: Implement registered metrics and decision object**

```python
@dataclass(frozen=True)
class AcceptanceInputs:
    p1_mae_ratio_to_m0: float
    p1_r2_drop_from_m0: float
    g1_mae_ratio_to_p1: float
    g1_rmse_ratio_to_p1: float
    g1_r2_drop_from_p1: float
    g1_fold_wins: int
    transfer_reduction_vs_f1: float
    transfer_reduction_vs_r1: float
    gate_median: float
    gate_p95: float


@dataclass(frozen=True)
class PhaseADecision:
    process_expert_credible: bool
    g1_noninferior: bool
    mean_improvement_path: bool
    transfer_safety_path: bool
    gate_collapsed_without_benefit: bool
    proceed_to_phase_b: bool
    path: str
    reasons: Sequence[str]
```

Implement `build_acceptance_inputs(summary_metrics, fold_metrics, negative_transfer_table, gate_values) -> AcceptanceInputs` and `assess_phase_a(inputs: AcceptanceInputs) -> PhaseADecision`. The builder, not the decision function, derives ratios and fold wins from saved tables; this keeps threshold logic directly unit-testable.

Use weighted MAE as primary, weighted RMSE and weighted R² as secondary. The decision must implement exactly:

- P1 credible when `MAE_P1 / MAE_M0 <= 1.05` and `R2_M0 - R2_P1 <= 0.02`.
- G1 noninferior when `MAE_G1 / MAE_P1 <= 1.01`, `RMSE_G1 / RMSE_P1 <= 1.03`, and `R2_G1 >= R2_P1`.
- Mean path when noninferior, `MAE_G1 / MAE_P1 <= 0.99`, and fold wins at least 3/5.
- Safety path when noninferior and G1 material-negative-transfer rate is at least 0.10 lower than both F1 and R1.
- Collapse when gate median `<0.05`, gate p95 `<0.20`, and neither path passes.
- `proceed_to_phase_b` only when P1 is credible, G1 is noninferior, at least one path passes, and collapse is false.

- [ ] **Step 5: Implement reporting outputs**

`write_phase_a_report` must create:

```text
outputs/sgrpn/phase_a/
  run_manifest.json
  audit/duration_audit.csv
  features/order_spectrum_cache.npz
  features/order_spectrum_cache.json
  folds/fold_<k>/seed_20260723/checkpoints_and_histories
  predictions/oof_predictions.csv
  evaluation/summary_metrics.csv
  evaluation/fold_metrics.csv
  evaluation/paired_bootstrap.csv
  evaluation/negative_transfer.csv
  evaluation/gate_statistics.csv
  evaluation/acceptance.json
  evaluation/method_notes.json
  evaluation/breakdowns/*.csv
  evaluation/figures/prediction_scatter.png
  evaluation/figures/residual_plot.png
  evaluation/figures/gate_distribution.png
  evaluation/figures/gate_condition_heatmap.png
```

`method_notes.json` must state that version is excluded from inputs, v3/v4 is a composite-domain stress test, Ch9/Ch10 orientation is uncertain, and Phase A contains one seed only.

- [ ] **Step 6: Test report reproducibility and old-output immutability**

Hash a fixture legacy directory, generate all reports from synthetic OOF rows, and assert hashes are unchanged. Delete the generated PNGs, regenerate from CSV/JSON inputs, and assert the expected files return without accessing checkpoints.

- [ ] **Step 7: Implement the CLI**

Expose:

```text
roughness-sgrpn audit --config configs/sgrpn_phase_a.yaml
roughness-sgrpn features --config configs/sgrpn_phase_a.yaml
roughness-sgrpn train-phase-a --config configs/sgrpn_phase_a.yaml [--fold 0] [--device auto] [--resume]
roughness-sgrpn evaluate-phase-a --config configs/sgrpn_phase_a.yaml
roughness-sgrpn run-phase-a --config configs/sgrpn_phase_a.yaml [--device auto] [--resume]
```

`run-phase-a` executes audit, cache construction, all five folds, and evaluation in that order. `--device auto` selects CUDA when `torch.cuda.is_available()` and otherwise CPU; record the selected device. Return nonzero on fingerprint mismatch, incomplete OOF coverage, leakage assertion, missing channels, non-finite arrays, or write attempt outside the new output directory.

- [ ] **Step 8: Run evaluation, reporting, and CLI tests**

Refresh only the existing project virtual environment's editable package metadata; do not resolve or download dependencies:

```powershell
python -m pip install -e . --no-deps
roughness-sgrpn --help
```

Then run:

```powershell
python -m pytest tests/sgrpn/test_evaluation.py tests/sgrpn/test_reporting.py tests/sgrpn/test_cli.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Run the complete existing and new test suite**

```powershell
python -m pytest -q
```

Expected: zero failures. Record test count and elapsed time in `outputs/sgrpn/phase_a/test_evidence.json`.

- [ ] **Step 10: Commit when Git is available**

```powershell
git add src/roughness/sgrpn/evaluation.py src/roughness/sgrpn/reporting.py src/roughness/sgrpn/cli.py tests/sgrpn/test_evaluation.py tests/sgrpn/test_reporting.py tests/sgrpn/test_cli.py
git commit -m "feat: add SGRPN phase A evaluation and CLI"
```

---

### Task 9: Run and verify the formal Phase A experiment

**Files:**
- Create through the CLI: `outputs/sgrpn/phase_a/**`
- Verify read-only: `outputs/scheme1/**`
- Verify read-only: `outputs/scheme1_physics/**`

**Interfaces:**
- Consumes: the fully tested CLI and frozen formal inputs.
- Produces: complete five-fold OOF evidence and the sole machine-readable Phase A decision.

- [ ] **Step 1: Capture legacy output hashes**

Use a read-only hashing helper from the new reporting module to write SHA-256 values for every existing file under `outputs/scheme1/` and `outputs/scheme1_physics/` to `outputs/sgrpn/phase_a/legacy_hashes_before.json`. Do not hash neural checkpoints by loading them; hash file bytes.

- [ ] **Step 2: Run the formal audit and cache build**

```powershell
roughness-sgrpn audit --config configs/sgrpn_phase_a.yaml
roughness-sgrpn features --config configs/sgrpn_phase_a.yaml
```

Expected: 212 groups, 586 segments, five outer folds, three channels, 25.6 kHz, and a recorded count of duration mismatches; no source file is modified.

- [ ] **Step 3: Execute the five outer folds**

```powershell
roughness-sgrpn train-phase-a --config configs/sgrpn_phase_a.yaml --device auto --resume
```

Expected: each fold completes P1/V1/F1/R1/G1 for seed 20260723, writes finite predictions, and records whether CUDA or CPU was used. Do not inspect an outer fold's metrics to alter later folds.

- [ ] **Step 4: Evaluate exactly once**

```powershell
roughness-sgrpn evaluate-phase-a --config configs/sgrpn_phase_a.yaml
```

Expected: `oof_predictions.csv` covers every segment exactly once per model; M0 is joined by `sample_id`; acceptance JSON contains every threshold, observed value, boolean result, reason, and `proceed_to_phase_b`.

- [ ] **Step 5: Verify artifacts and legacy immutability**

Recompute legacy hashes into `legacy_hashes_after.json` and require byte-for-byte equality with the before file. Run:

```powershell
python -m pytest tests/sgrpn -q
```

Expected: zero failures. Also require no missing OOF row, no duplicate `(sample_id, model)`, finite metrics, gates in `[0,1]`, and exactly 10,000 bootstrap repetitions for each registered comparison.

- [ ] **Step 6: Apply the stopping rule without reinterpretation**

Read `outputs/sgrpn/phase_a/evaluation/acceptance.json`:

- If `proceed_to_phase_b` is `false`, stop model development, summarize the registered failure reason, retain all outputs, and do not write or execute a Phase B plan.
- If `proceed_to_phase_b` is `true`, write `docs/superpowers/plans/2026-08-21-sgrpn-phase-b-implementation.md` from Sections 8–13 of the approved spec before touching Phase B code.

- [ ] **Step 7: Commit formal evidence when Git is available**

Do not commit binary checkpoints or the large `.npz` cache. If a valid repository exists and its ignore policy permits experiment summaries, commit only configuration, source, tests, compact CSV/JSON evaluation files, and figure files explicitly requested by the user.

---

## Execution Checkpoints

1. After Task 3: verify the cache fingerprint and known-order tests before any model work.
2. After Task 6: inspect one outer-train OOF assignment table and prove every row was predicted once by an unseen-group P1.
3. After Task 7: inspect expert state hashes before/after gate training and one synthetic full-fold run.
4. After Task 8: require the full suite to pass before formal data execution.
5. After Task 9: report the acceptance decision exactly as emitted; do not tune after outer evaluation.
