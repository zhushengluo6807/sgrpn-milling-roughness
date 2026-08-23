# SGRPN Phase B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and formally evaluate the three-seed, repeated-measure-aware SGRPN probability pipeline with leakage-safe heteroscedastic and homoscedastic Gaussian scales plus 90%/95% group-conformal intervals, without changing any Phase A or legacy artifact.

**Architecture:** Treat the fingerprinted Phase A design, inputs, cache, folds, and passing acceptance decision as immutable prerequisites, then retrain the fixed P1→Residual-CNN→G1 mean path independently for seeds `20260723`, `20260724`, and `20260725`. Freeze each fitted mean model, train an 82-input conditional scale head and a global-scale ablation on the three readings with region-preserving weights, obtain calibration scores only from group-disjoint inner OOF mean/scale predictions, and evaluate outer OOF Gaussian and conformal predictions from saved artifacts.

**Tech Stack:** Python 3.12.13, PyTorch 2.11.0, NumPy 2.5.1, pandas 2.3.3, scikit-learn 1.9.0, PyYAML, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-sgrpn-design.md` (approved Sections 8–13; Phase A contracts in Sections 2–7 remain binding inputs)

## Global Constraints

- Before any Phase B source, test, configuration, training, evaluation, or output action, require `outputs/sgrpn/phase_a/evaluation/acceptance.json` to be byte-hash registered by the Phase A `run_manifest.json`, require both top-level and nested `decision.proceed_to_phase_b` to be `true`, and require `phase_b_executed` to be `false`.
- Recompute and validate the Phase A configuration, manifest, folds, window-index, M0, cache, training fingerprint, five fold completion markers, and every registered Phase A fold artifact. Every marker, state file, checkpoint, history, and scaler metadata record must declare exact protocol `sgrpn-phase-a-v2`; any missing protocol, `sgrpn-phase-a-v1`, other protocol, artifact-hash mismatch, or fingerprint mismatch stops Phase B before `outputs/sgrpn/phase_b/` or any child/temp directory is created.
- Treat `configs/sgrpn_phase_a.yaml`, `outputs/sgrpn/phase_a/**`, `outputs/scheme1/**`, and `outputs/scheme1_physics/**` as immutable inputs. Write Phase B runtime artifacts only under `outputs/sgrpn/phase_b/`.
- Use only the 586 registered segments, 212 registered `group_id` values, three registered Ra readings per segment, fixed five outer folds, fixed four inner group folds, cached 3×361 order spectra, seven quality features, and `1/split_count` region weights. Do not acquire, create, remove, relabel, or substitute samples, groups, readings, windows, folds, or signal versions.
- The formal run is the approved Phase B extension, not a new experiment: do not add machining, idle, shutdown, repeat-measurement, synthetic-label, hyperparameter-search, alternative-split, or post-result follow-up runs.
- Run Python with `E:\CodeX\机床项目\.venv\Scripts\python.exe`; do not mutate `D:\CodexPython`, install packages, or change the frozen environment.
- Phase B seeds are exactly `20260723`, `20260724`, and `20260725`. Retrain the fixed mean model independently for every seed; do not initialize from the Phase A seed checkpoint and do not select favorable seeds, folds, epochs, or results after outer evaluation.
- Preserve the Phase A mean architecture, order grid, channel handling, process features, quality features, weighted Huber loss, optimizer settings, four-fold epoch selection, and no-joint-fine-tuning rule. Phase B does not reopen the Phase A model matrix, thresholds, gate architecture, order range, or hyperparameters.
- Keep every `group_id` wholly within one side of each outer or inner split. No standardizer, early-stopping decision, residual target, variance fit, conformal score, conformal quantile, or report selector may read an outer-test label.
- Each segment remains one region sample with readings `[ra_1, ra_2, ra_3]`; the mean target is their arithmetic mean. Average probability loss across the three readings before multiplying by `1/split_count`, so repeating the labels never triples a region's total weight.
- Treat the three reading terms as repeated observations sharing one region-level `mu` and one region-level `sigma`, not as three independent samples or three separately tunable task heads.
- The heteroscedastic head input is exactly `[process_9, vibration_embedding_64, quality_7, gate_1, abs(prediction - process_mean)_1]`, totaling 82 values. For swap-ensemble validation/inference, `prediction`, `process_mean`, `embedding`, and `gate` are the fields returned by `average_swap_predictions`, so the correction is the actual ensemble correction `abs(mean(g * residual))`, never the algebraically different `abs(mean(g) * mean(residual))`. Its output is exactly `softplus(raw_scale) + 1e-4`.
- Freeze P1, Residual-CNN, and Trust-Gate parameters and training-state buffers while fitting either scale model. Compare every mean-model state tensor before and after scale fitting with `torch.equal` and fail on any change.
- The homoscedastic ablation learns one global scalar `sigma` per outer fold and seed through the same repeated-reading Gaussian NLL, optimizer, split, selection, refit, and calibration protocol as the heteroscedastic head.
- Calibrate 90% and 95% intervals separately for each `(outer_fold, seed, scale_model)`. Each calibration `group_id` contributes one score: the maximum standardized absolute error over all its registered regions and all three readings.
- Use finite-sample order statistic `k = ceil((m + 1) * (1 - alpha))`, one-based, and `q_alpha = sorted_group_scores[k - 1]` for `alpha ∈ {0.10, 0.05}`. Reject `k > m`; do not interpolate quantiles or treat regions/readings as independent calibration units.
- Raw Gaussian intervals use fixed standard-normal quantiles `1.6448536269514722` for 90% and `1.959963984540054` for 95%. Conformal intervals use `mu ± q_alpha * sigma`.
- Report mean-target MAE/RMSE/R², repeated-reading Gaussian NLL and Gaussian CRPS, single-reading coverage, `group_id` simultaneous coverage, average width, Winkler score, raw-versus-conformal results, and heteroscedastic-versus-homoscedastic ablation for every seed and in an all-seed summary.
- Interpret learned `sigma` only as conditional predictive dispersion combining repeat variation and unmodeled error. Seed-to-seed variance is a descriptive model-instability proxy, not a posterior epistemic variance; do not add it to `sigma²` to create unregistered intervals or claim an aleatoric/epistemic decomposition.
- Main coverage language is simultaneous coverage for a new exchangeable `group_id` over all existing-type regions and repeats. Single-reading coverage is secondary; v3/v4 is only a composite-domain stress test confounded with the speed grid.
- Use exactly 10,000 paired bootstrap repetitions with resampling unit `group_id` and preserve all seeds, folds, failures, stops, and unfavorable ablations.
- If the heteroscedastic head does not strictly improve registered Winkler interval score over the homoscedastic ablation, the paper decision must set `emphasize_heteroscedasticity=false`. Because the approved spec registers no numeric practical-width threshold, report widths and `practical_width_threshold_registered=false`; do not invent a threshold or rerun based on width.
- All figures and paper tables must regenerate from saved CSV/JSON predictions, scores, quantiles, configuration, and method-note artifacts without loading checkpoints or relying on in-memory state.

## File Map

| Path | Responsibility |
|---|---|
| `configs/sgrpn_phase_b.yaml` | Freeze Phase B gate paths, three seeds, scale optimization, conformal levels, and output root |
| `src/roughness/sgrpn/config.py` | Add the isolated Phase B configuration and immutable Phase A handoff validator |
| `src/roughness/sgrpn/data.py` | Validate and expose the registered `[ra_1, ra_2, ra_3]` matrix without expanding rows |
| `src/roughness/sgrpn/models.py` | Add the 82→32→16→1 heteroscedastic head, global-scale ablation, and repeated Gaussian NLL |
| `src/roughness/sgrpn/probability.py` | Build frozen-mean scale features, Gaussian CRPS/raw intervals, group conformal scores/quantiles, and interval metrics |
| `src/roughness/sgrpn/training.py` | Generalize the fixed G1 mean path to registered Phase B seeds while preserving Phase A behavior |
| `src/roughness/sgrpn/phase_b_training.py` | Orchestrate nested mean/scale OOF calibration, outer refits, checkpoints, fingerprints, resume, and OOF prediction rows |
| `src/roughness/sgrpn/evaluation.py` | Validate the Phase B OOF Cartesian product and build probability/coverage/ablation tables and paper decision |
| `src/roughness/sgrpn/reporting.py` | Persist reproducible Phase B tables, method limits, figures, integrity manifest, and immutable-input proof |
| `src/roughness/sgrpn/cli.py` | Add `preflight-phase-b`, `train-phase-b`, `evaluate-phase-b`, and `run-phase-b` commands |
| `tests/sgrpn/test_config.py` | Phase B gate, config, hash, and output-root tests |
| `tests/sgrpn/test_data.py` | Three-reading and region-weight contract tests |
| `tests/sgrpn/test_models.py` | Scale architecture, positivity, frozen-state, and NLL tests |
| `tests/sgrpn/test_probability.py` | CRPS, intervals, conformal order statistic, group aggregation, coverage, width, and Winkler tests |
| `tests/sgrpn/test_training.py` | Fixed mean-path generalization and Phase A regression tests |
| `tests/sgrpn/test_phase_b_training.py` | Nested leakage barriers, three-seed fold orchestration, resume, and synthetic integration tests |
| `tests/sgrpn/test_evaluation.py` | Probability OOF schema, metrics, ablation, and paper stopping-rule tests |
| `tests/sgrpn/test_reporting.py` | Phase B artifact reconstruction and immutable-input tests |
| `tests/sgrpn/test_cli.py` | Phase B command ordering, failure codes, and formal-run barriers |

## Registered Interfaces and Schemas

Use these names consistently in every task:

```python
PHASE_B_SEEDS = (20260723, 20260724, 20260725)
SCALE_MODELS = ("heteroscedastic", "homoscedastic")
ALPHAS = (0.10, 0.05)

PHASE_B_PREDICTION_COLUMNS = (
    "sample_id", "group_id", "version", "fold", "seed", "scale_model",
    "target_mean", "ra_1", "ra_2", "ra_3", "sample_weight",
    "mu", "sigma", "gate", "correction",
    "raw_lower_90", "raw_upper_90", "raw_lower_95", "raw_upper_95",
    "conformal_q_90", "conformal_lower_90", "conformal_upper_90",
    "conformal_q_95", "conformal_lower_95", "conformal_upper_95",
)

PHASE_B_MEAN_COLUMNS = (
    "sample_id", "group_id", "version", "fold", "seed", "model",
    "target_mean", "prediction", "sample_weight", "process_mean",
    "residual", "gate",
)

CALIBRATION_SCORE_COLUMNS = (
    "group_id", "outer_fold", "inner_fold", "seed", "scale_model",
    "score", "region_count", "reading_count",
)
```

The final public signatures are:

```python
# load_phase_b_config(path: str | Path) -> PhaseBConfig
# validate_phase_b_handoff(config: PhaseBConfig) -> PhaseAHandoff
# validate_phase_b_output_root(path: str | Path, *, output_root: str | Path | None = None) -> Path
# repeat_measure_batch(frame: pd.DataFrame) -> RepeatMeasureBatch
# build_scale_features(output: ModelOutput, process: Tensor, quality: Tensor) -> Tensor
# repeated_gaussian_nll(mu: Tensor, sigma: Tensor, readings: Tensor, weight: Tensor) -> Tensor
# freeze_mean_model(model: SelectiveGatedModel) -> dict[str, Tensor]
# assert_mean_model_unchanged(model: SelectiveGatedModel, snapshot: Mapping[str, Tensor]) -> None
# gaussian_crps(mu: Any, sigma: Any, readings: Any, weight: Any) -> float
# raw_gaussian_interval(mu: Any, sigma: Any, *, alpha: float) -> tuple[np.ndarray, np.ndarray]
# group_conformal_scores(group_ids: Any, mu: Any, sigma: Any, readings: Any) -> pd.DataFrame
# finite_sample_group_quantile(scores: Any, *, alpha: float) -> GroupConformalResult
# conformal_interval(mu: Any, sigma: Any, quantile: float) -> tuple[np.ndarray, np.ndarray]
# single_reading_coverage(lower: Any, upper: Any, readings: Any, weight: Any) -> float
# simultaneous_group_coverage(group_ids: Any, lower: Any, upper: Any, readings: Any) -> float
# mean_interval_width(lower: Any, upper: Any, weight: Any) -> float
# winkler_score(lower: Any, upper: Any, readings: Any, weight: Any, *, alpha: float) -> float
# fit_g1_mean_path(config: SGRPNConfig, bundle: DataBundle, cache: OrderSpectrumCache, fold: int, seed: int, train_sample_ids: Sequence[str], *, device: str | torch.device, backend: TrainingBackend | None = None, batch_size: int = 8) -> MeanPathArtifacts
# fit_scale_model(*, mean_model: SelectiveGatedModel, scale_model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, max_epochs: int, patience: int, learning_rate: float, weight_decay: float, device: torch.device, seed: int) -> ScaleFit
# build_nested_calibration(config: PhaseBConfig, phase_a_config: SGRPNConfig, bundle: DataBundle, cache: OrderSpectrumCache, fold: int, seed: int, *, device: str | torch.device, backend: TrainingBackend | None = None, batch_size: int = 8) -> Mapping[str, CalibrationArtifacts]
# run_phase_b_fold(config: PhaseBConfig, handoff: PhaseAHandoff, phase_a_config: SGRPNConfig, bundle: DataBundle, cache: OrderSpectrumCache, fold: int, seed: int, *, device: str | torch.device | None = None, backend: TrainingBackend | None = None, batch_size: int = 8, output_root: str | Path | None = None) -> PhaseBFoldArtifacts
# run_phase_b(config: PhaseBConfig, handoff: PhaseAHandoff, phase_a_config: SGRPNConfig, bundle: DataBundle, cache: OrderSpectrumCache, *, device: str | torch.device | None = None, backend: TrainingBackend | None = None, batch_size: int = 8, output_root: str | Path | None = None) -> PhaseBRunArtifacts
# validate_phase_b_prediction_cartesian(predictions: pd.DataFrame, *, manifest: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame
# probability_metric_table(predictions: pd.DataFrame) -> pd.DataFrame
# seed_uncertainty_summary(predictions: pd.DataFrame) -> pd.DataFrame
# paired_probability_bootstrap(predictions: pd.DataFrame, *, metric: str, repetitions: int = 10000, seed: int = 20260723) -> BootstrapResult
# assess_phase_b_claims(probability_metrics: pd.DataFrame) -> PhaseBPaperDecision
# write_phase_b_report(config: PhaseBConfig, handoff: PhaseAHandoff, bundle: DataBundle, mean_predictions: pd.DataFrame, probability_predictions: pd.DataFrame, calibration_scores: pd.DataFrame, quantiles: pd.DataFrame) -> Mapping[str, Path]
# validate_phase_b_outputs(config: PhaseBConfig) -> None
```

The Phase B probability OOF table must contain exactly `586 * 3 * 2 = 3516` rows, one row per registered `(sample_id, seed, scale_model)`. The companion fixed-mean table must contain exactly `586 * 3 * 3 = 5274` P1/R1/G1 rows so registered mean accuracy, gate, residual, Bootstrap, and negative-transfer outputs can be rebuilt without Phase A predictions. Within a probability row, one `mu` and `sigma` apply to all three readings; every interval bound is finite and ordered, and the conformal quantile is constant within `(fold, seed, scale_model, coverage)`.

---

### Task 1: Enforce the immutable Phase A gate and freeze Phase B configuration

**Files:**
- Create: `configs/sgrpn_phase_b.yaml`
- Modify: `src/roughness/sgrpn/config.py`
- Modify: `tests/sgrpn/test_config.py`

**Interfaces:**
- Consumes: `load_sgrpn_config(path) -> SGRPNConfig`, `load_data_bundle`, `load_order_cache`, the current `build_run_fingerprint`, the existing Phase A deep artifact validators, Phase A `acceptance.json`, `run_manifest.json`, cache metadata, five `complete.json` markers, five `state.json` files, and all registered checkpoint/history/scaler byte hashes and embedded metadata.
- Produces: `PhaseBConfig`, `PhaseAHandoff`, `load_phase_b_config(path: str | Path) -> PhaseBConfig`, `validate_phase_b_handoff(config: PhaseBConfig) -> PhaseAHandoff`, and `validate_phase_b_output_root(path: str | Path, *, output_root: str | Path | None = None) -> Path`.

- [ ] **Step 1: Run the read-only gate preflight before editing any Phase B file**

Run from the worktree root:

```powershell
$a = Get-Content -Raw -LiteralPath outputs/sgrpn/phase_a/evaluation/acceptance.json | ConvertFrom-Json
$m = Get-Content -Raw -LiteralPath outputs/sgrpn/phase_a/run_manifest.json | ConvertFrom-Json
$expectedProtocol = 'sgrpn-phase-a-v2'
if (-not $a.proceed_to_phase_b -or -not $a.decision.proceed_to_phase_b -or $a.phase_b_executed) { throw 'Phase A gate is closed' }
$acceptanceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath outputs/sgrpn/phase_a/evaluation/acceptance.json).Hash.ToLowerInvariant()
if ($acceptanceHash -ne $m.artifacts.'evaluation/acceptance.json') { throw 'Phase A acceptance fingerprint mismatch' }
0..4 | ForEach-Object {
    $marker = Get-Content -Raw -LiteralPath "outputs/sgrpn/phase_a/folds/fold_$_/seed_20260723/complete.json" | ConvertFrom-Json
    $state = Get-Content -Raw -LiteralPath "outputs/sgrpn/phase_a/folds/fold_$_/seed_20260723/state.json" | ConvertFrom-Json
    if ($marker.protocol -cne $expectedProtocol -or $marker.status -ne 'complete' -or $marker.fingerprint -ne $m.training_fingerprint -or $marker.seed -ne 20260723 -or $marker.fold -ne $_) { throw "Phase A fold $_ marker handoff mismatch" }
    if ($state.protocol -cne $expectedProtocol -or $state.status -ne 'complete' -or $state.fingerprint -ne $m.training_fingerprint -or $state.seed -ne 20260723 -or $state.fold -ne $_) { throw "Phase A fold $_ state handoff mismatch" }
}
Write-Output "Phase A gate verified: $($a.decision.path)"
```

Expected: exit 0 and `Phase A gate verified: transfer_safety`. This shell check is only the first read-only guard; Step 7 must still recompute scientific identity and run the deep Python validators. On any other result, stop execution without creating or modifying Phase B files or directories.

- [ ] **Step 2: Write failing Phase B config tests**

```python
def test_phase_b_protocol_is_exactly_registered():
    config = load_phase_b_config("configs/sgrpn_phase_b.yaml")
    assert tuple(config.seeds) == (20260723, 20260724, 20260725)
    assert tuple(config.alphas) == (0.10, 0.05)
    assert config.inner_splits == 4
    assert config.max_epochs == 200
    assert config.patience == 20
    assert config.variance_learning_rate == 1e-3
    assert config.weight_decay == 1e-4
    assert config.output_dir.name == "phase_b"


def test_phase_b_rejects_unregistered_seed(tmp_path):
    source = Path("configs/sgrpn_phase_b.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["seeds"] = [20260723, 7, 20260725]
    path = tmp_path / "bad_phase_b.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 20260723, 20260724, 20260725"):
        load_phase_b_config(path)
```

- [ ] **Step 3: Run the config tests to verify RED**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_config.py -k phase_b -v
```

Expected: FAIL because `PhaseBConfig` and `load_phase_b_config` do not exist.

- [ ] **Step 4: Add the exact Phase B dataclasses**

```python
@dataclass(frozen=True)
class PhaseBConfig:
    phase_a_config_path: Path
    phase_a_acceptance_path: Path
    phase_a_run_manifest_path: Path
    output_dir: Path
    seeds: Sequence[int]
    alphas: Sequence[float]
    inner_splits: int
    max_epochs: int
    patience: int
    variance_learning_rate: float
    weight_decay: float
    bootstrap_repetitions: int


@dataclass(frozen=True)
class PhaseAHandoff:
    acceptance_sha256: str
    run_manifest_sha256: str
    phase_a_config_sha256: str
    training_fingerprint: str
    cache_sha256: str
    input_sha256: Mapping[str, str]
```

`load_phase_b_config` must reject extra/missing keys, booleans masquerading as numbers, a noncanonical seed order, alphas other than `(0.10, 0.05)`, a value other than four inner splits, and an output root other than exact project `outputs/sgrpn/phase_b` unless a test passes an explicit exact `output_root`.

- [ ] **Step 5: Write the frozen YAML**

```yaml
phase_a_config_path: sgrpn_phase_a.yaml
phase_a_acceptance_path: ../outputs/sgrpn/phase_a/evaluation/acceptance.json
phase_a_run_manifest_path: ../outputs/sgrpn/phase_a/run_manifest.json
output_dir: ../outputs/sgrpn/phase_b
seeds: [20260723, 20260724, 20260725]
alphas: [0.10, 0.05]
inner_splits: 4
max_epochs: 200
patience: 20
variance_learning_rate: 0.001
weight_decay: 0.0001
bootstrap_repetitions: 10000
```

The variance optimizer values deliberately reuse the fixed Phase A gate optimizer and training limits; they are configuration constants, not values selected from Phase B results.

- [ ] **Step 6: Write failing handoff-tamper tests**

Build one valid minimal Phase A v2 fixture whose marker registers the exact hashes of its checkpoint, history, and scaler. Parameterize these fail-closed mutations: remove `protocol` from marker or state; set either to `sgrpn-phase-a-v1`; remove or change embedded `protocol` in a checkpoint, history row, or scaler `metadata_json`; change any embedded fingerprint; change a registered artifact byte without updating its marker hash; change a marker artifact hash; and change the completion fingerprint. Also mutate one byte in each of acceptance, Phase A config, manifest, folds, and cache NPZ. For every case, assert `validate_phase_b_handoff` raises `ValueError` naming the affected artifact and that neither `output_dir` nor any child/temp path exists:

```python
@pytest.mark.parametrize(
    ("artifact", "mutation"),
    [
        ("marker", "missing_protocol"),
        ("marker", "v1_protocol"),
        ("state", "missing_protocol"),
        ("state", "v1_protocol"),
        ("checkpoint", "missing_protocol"),
        ("checkpoint", "v1_protocol"),
        ("checkpoint", "protocol_mismatch"),
        ("checkpoint", "fingerprint_mismatch"),
        ("history", "missing_protocol"),
        ("history", "v1_protocol"),
        ("history", "protocol_mismatch"),
        ("history", "fingerprint_mismatch"),
        ("scaler", "missing_protocol"),
        ("scaler", "v1_protocol"),
        ("scaler", "protocol_mismatch"),
        ("scaler", "fingerprint_mismatch"),
        ("checkpoint", "registered_hash_mismatch"),
        ("history", "registered_hash_mismatch"),
        ("scaler", "registered_hash_mismatch"),
        ("marker", "fingerprint_mismatch"),
    ],
)
def test_phase_b_handoff_rejects_deep_phase_a_mismatch_without_output(
    valid_phase_a_v2_fixture, artifact, mutation
):
    config, output_dir = valid_phase_a_v2_fixture
    mutate_phase_a_fixture(config, artifact=artifact, mutation=mutation)
    with pytest.raises(ValueError, match=artifact):
        validate_phase_b_handoff(config)
    assert not output_dir.exists()
```

The fixture helper must rewrite hashes only while constructing its original valid state; mutation helpers must not repair the marker after tampering. Add a positive test proving a fully consistent `sgrpn-phase-a-v2` fixture returns the recomputed Phase A fingerprint and still does not create the Phase B directory. Add a separate closed-acceptance test with the same no-directory assertion.

- [ ] **Step 7: Implement the handoff validator**

`validate_phase_b_handoff` is read-only and must run before `Path.mkdir`, temporary-file creation, writer construction, or any other Phase B output action. It must verify, in order: acceptance schema and both true gate fields; `subjective_override_allowed is false`; `phase_b_executed is false`; acceptance byte hash against the run manifest; Phase A config byte hash; every `input_sha256` source; and feature-cache NPZ against its cache JSON. It must then load the current Phase A config, canonical bundle, and cache, call the current `build_run_fingerprint(config, bundle, cache)`, and compare every component hash plus `.value` with the Phase A run manifest instead of trusting the manifest's `training_fingerprint` field.

For each of the five folds, require exact protocol `sgrpn-phase-a-v2` on both `complete.json` and `state.json`, exact seed `20260723`, fold number, complete five-model state, and the recomputed fingerprint. Delegate checkpoint/history/scaler validation to the existing Phase A deep validators (`_validate_completed_artifacts` and its `_validate_checkpoint`, `_validate_history`, and `_validate_scalers` chain), so every marker-registered byte hash, embedded protocol, embedded fingerprint, schema, epoch metadata, finite tensor/scale constraint, and G1-embedded P1/R1 state is rechecked from bytes. Reuse `_load_and_validate_persisted_oof` for each fold's prediction hash and row identity. Use function-local imports if `config.py` would otherwise create a module-import cycle; do not duplicate or weaken the Phase A validation logic. Missing protocol, a v1 protocol, any hash mismatch, or any fingerprint mismatch must raise before the Phase B output root exists. The function returns only recomputed hashes and IDs, never loaded model parameters.

- [ ] **Step 8: Run config and handoff tests to verify GREEN**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_config.py -v
```

Expected: all config tests pass; the positive v2 fixture returns the recomputed fingerprint; every missing/v1/deep protocol, hash, or fingerprint mutation fails closed; and no success or failure case creates a `phase_b` directory.

- [ ] **Step 9: Commit the gate and protocol**

```powershell
git add configs/sgrpn_phase_b.yaml src/roughness/sgrpn/config.py tests/sgrpn/test_config.py
git commit -m "feat: lock SGRPN phase B gate"
```

---

### Task 2: Preserve the three-reading region contract

**Files:**
- Modify: `src/roughness/sgrpn/data.py`
- Modify: `tests/sgrpn/test_data.py`

**Interfaces:**
- Consumes: a manifest frame with `ra_1`, `ra_2`, `ra_3`, `ra_mean`, `sample_weight`, and `split_count`.
- Produces: `RepeatMeasureBatch(readings: np.ndarray, mean: np.ndarray, weight: np.ndarray)` and `repeat_measure_batch(frame: pd.DataFrame) -> RepeatMeasureBatch`.

- [ ] **Step 1: Write failing shape and mean tests**

```python
def test_repeat_measure_batch_keeps_one_row_per_region():
    frame = pd.DataFrame({
        "ra_1": [0.4, 0.7], "ra_2": [0.5, 0.8], "ra_3": [0.6, 0.9],
        "ra_mean": [0.5, 0.8], "sample_weight": [0.5, 0.25],
        "split_count": [2, 4],
    })
    batch = repeat_measure_batch(frame)
    assert batch.readings.shape == (2, 3)
    np.testing.assert_allclose(batch.mean, [0.5, 0.8])
    np.testing.assert_allclose(batch.weight, [0.5, 0.25])
```

- [ ] **Step 2: Write the region-total-weight failure test**

```python
def test_repeat_measure_batch_rejects_noncanonical_region_weight():
    frame = pd.DataFrame({
        "ra_1": [0.4], "ra_2": [0.5], "ra_3": [0.6], "ra_mean": [0.5],
        "sample_weight": [1.0], "split_count": [2],
    })
    with pytest.raises(ValueError, match="1/split_count"):
        repeat_measure_batch(frame)
```

- [ ] **Step 3: Run the targeted tests to verify RED**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_data.py -k repeat_measure -v
```

Expected: FAIL because the new dataclass/function is absent.

- [ ] **Step 4: Implement the exact validator**

```python
@dataclass(frozen=True)
class RepeatMeasureBatch:
    readings: np.ndarray  # [N, 3]
    mean: np.ndarray      # [N]
    weight: np.ndarray    # [N]


def repeat_measure_batch(frame: pd.DataFrame) -> RepeatMeasureBatch:
    readings = frame.loc[:, ["ra_1", "ra_2", "ra_3"]].to_numpy(np.float64)
    mean = frame["ra_mean"].to_numpy(np.float64)
    weight = frame["sample_weight"].to_numpy(np.float64)
    split_count = frame["split_count"].to_numpy(np.float64)
    # Require finite values, shape [N, 3], positive split_count,
    # exact registered weight within atol=1e-12, and stored mean within atol=1e-12.
```

Return one row per input region. Never melt or repeat the frame. Reject missing/nonfinite readings and reject any stored `ra_mean` inconsistent with the arithmetic mean.

- [ ] **Step 5: Run data tests to verify GREEN**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_data.py -v
```

Expected: all data tests pass, including existing Phase A duration and split checks.

- [ ] **Step 6: Commit the repeat-measure contract**

```powershell
git add src/roughness/sgrpn/data.py tests/sgrpn/test_data.py
git commit -m "feat: preserve SGRPN repeat measurements"
```

---

### Task 3: Add frozen-mean Gaussian scale models and loss

**Files:**
- Modify: `src/roughness/sgrpn/models.py`
- Modify: `tests/sgrpn/test_models.py`

**Interfaces:**
- Consumes: frozen `ModelOutput` from G1, scaled process `[B,9]`, scaled quality `[B,7]`, and readings `[B,3]`.
- Produces: `VarianceHead`, `GlobalScale`, `build_scale_features(output: ModelOutput, process: Tensor, quality: Tensor) -> Tensor`, `repeated_gaussian_nll(mu: Tensor, sigma: Tensor, readings: Tensor, weight: Tensor) -> Tensor`, `freeze_mean_model(model: SelectiveGatedModel) -> dict[str, Tensor]`, and `assert_mean_model_unchanged(model, snapshot) -> None`.

- [ ] **Step 1: Write the failing 82-input and positivity tests**

```python
def test_scale_features_are_exactly_registered_82_values():
    output = ModelOutput(
        prediction=torch.tensor([1.1]), process_mean=torch.tensor([1.0]),
        residual=torch.tensor([-0.4]),
        gate=torch.tensor([0.25]), embedding=torch.zeros(1, 64),
    )
    features = build_scale_features(output, torch.zeros(1, 9), torch.zeros(1, 7))
    assert features.shape == (1, 82)
    assert features[0, -2].item() == pytest.approx(0.25)
    assert features[0, -1].item() == pytest.approx(0.10)


def test_variance_head_is_finite_and_bounded_away_from_zero():
    sigma = VarianceHead()(torch.zeros(4, 82))
    assert sigma.shape == (4,)
    assert torch.isfinite(sigma).all()
    assert torch.all(sigma >= 1e-4)
```

- [ ] **Step 2: Write the exact repeated-NLL weight test**

```python
def test_repeated_nll_averages_reads_before_region_weighting():
    mu = torch.tensor([0.0, 1.0])
    sigma = torch.ones(2)
    readings = torch.tensor([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    weight = torch.tensor([0.5, 0.25])
    per_read = 0.5 * math.log(2.0 * math.pi) + 0.5 * (readings - mu[:, None]).square()
    expected = (weight * per_read.mean(dim=1)).sum() / weight.sum()
    assert repeated_gaussian_nll(mu, sigma, readings, weight) == pytest.approx(expected)
```

- [ ] **Step 3: Run model tests to verify RED**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_models.py -k "scale or repeated_nll or mean_model" -v
```

Expected: FAIL on missing scale interfaces.

- [ ] **Step 4: Implement the fixed architectures**

```python
class VarianceHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(82, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        _require_finite_matrix(features, name="scale_features", width=82)
        return F.softplus(self.network(features).squeeze(1)) + 1e-4


class GlobalScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw_scale = nn.Parameter(torch.zeros(()))

    def forward(self, features: Tensor) -> Tensor:
        _require_finite_matrix(features, name="scale_features", width=82)
        return (F.softplus(self.raw_scale) + 1e-4).expand(features.shape[0])
```

- [ ] **Step 5: Implement feature assembly and repeated NLL**

`build_scale_features` must require populated finite `prediction`, `process_mean`, `embedding`, and `gate`, validate matching batch sizes and gate range, calculate `correction_abs = abs(prediction - process_mean)`, and concatenate in the registered order. For a single orientation this equals `abs(gate * residual)`; for a swap ensemble it equals the absolute averaged correction and must not be reconstructed from separately averaged gate/residual fields. Add an anti-correlated two-orientation regression in which `mean(g) * mean(residual)` differs from `mean(g * residual)` and require the feature to equal `abs(averaged_prediction - averaged_process_mean)`. `repeated_gaussian_nll` must require shapes `[B]`, `[B]`, `[B,3]`, `[B]`, positive finite sigma/weights, compute `0.5*log(2*pi*sigma²) + 0.5*((y-mu)/sigma)²`, average axis 1, then normalize the region-weighted sum by `weight.sum()`.

- [ ] **Step 6: Implement and test immutable mean-state snapshots**

Snapshot every `state_dict()` tensor with `detach().cpu().clone()`, set every mean-model parameter `requires_grad_(False)`, call `eval()`, and compare key sets, dtypes, shapes, and values with `torch.equal`. Include BatchNorm buffers so scale training cannot silently alter the encoder.

- [ ] **Step 7: Run all model tests to verify GREEN**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_models.py -v
```

Expected: all model tests pass, including Phase A swap invariance and gate fallback tests.

- [ ] **Step 8: Commit the probability heads**

```powershell
git add src/roughness/sgrpn/models.py tests/sgrpn/test_models.py
git commit -m "feat: add repeated Gaussian scale heads"
```

---

### Task 4: Implement Gaussian, conformal, coverage, sharpness, NLL, and CRPS primitives

**Files:**
- Create: `src/roughness/sgrpn/probability.py`
- Create: `tests/sgrpn/test_probability.py`

**Interfaces:**
- Consumes: finite per-region `mu`, positive `sigma`, three readings, region weights, group IDs, and alpha.
- Produces: `GroupConformalResult`, `gaussian_crps`, `raw_gaussian_interval`, `group_conformal_scores`, `finite_sample_group_quantile`, `conformal_interval`, `single_reading_coverage`, `simultaneous_group_coverage`, `mean_interval_width`, and `winkler_score`.

- [ ] **Step 1: Write failing Gaussian CRPS and raw-interval tests**

```python
def test_gaussian_crps_at_its_mean():
    value = gaussian_crps(
        np.array([0.0]), np.array([2.0]), np.array([[0.0, 0.0, 0.0]]), np.array([1.0])
    )
    assert value == pytest.approx(2.0 * (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi))


def test_raw_gaussian_interval_uses_registered_normal_quantile():
    lower, upper = raw_gaussian_interval(np.array([1.0]), np.array([2.0]), alpha=0.10)
    np.testing.assert_allclose(lower, [1.0 - 2.0 * 1.6448536269514722])
    np.testing.assert_allclose(upper, [1.0 + 2.0 * 1.6448536269514722])
```

- [ ] **Step 2: Write failing group-score and finite-sample tests**

```python
def test_each_group_contributes_one_max_score():
    scores = group_conformal_scores(
        group_ids=np.array(["g1", "g1", "g2"]),
        mu=np.array([0.0, 1.0, 0.0]), sigma=np.ones(3),
        readings=np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]),
    )
    assert scores["group_id"].tolist() == ["g1", "g2"]
    np.testing.assert_allclose(scores["score"], [3.0, 0.5])


def test_conformal_quantile_uses_ceil_m_plus_one_without_interpolation():
    result = finite_sample_group_quantile(np.arange(1.0, 20.0), alpha=0.10)
    assert result.group_count == 19
    assert result.order_index == 18
    assert result.quantile == 18.0
```

- [ ] **Step 3: Run probability tests to verify RED**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_probability.py -v
```

Expected: collection FAIL because `roughness.sgrpn.probability` is absent.

- [ ] **Step 4: Implement exact result and interval functions**

```python
@dataclass(frozen=True)
class GroupConformalResult:
    alpha: float
    group_count: int
    order_index: int  # one-based k
    quantile: float


def finite_sample_group_quantile(scores: Any, *, alpha: float) -> GroupConformalResult:
    values = np.sort(np.asarray(scores, dtype=np.float64))
    k = math.ceil((len(values) + 1) * (1.0 - alpha))
    if k > len(values):
        raise ValueError("insufficient calibration groups for requested alpha")
    return GroupConformalResult(alpha, len(values), k, float(values[k - 1]))
```

Accept only alpha `0.10` or `0.05`. Reject missing groups, nonpositive scales, nonfinite values, empty arrays, and group-score tables with duplicates.

- [ ] **Step 5: Implement Gaussian CRPS over repeated readings**

Use `z = (reading - mu[:, None]) / sigma[:, None]` and
`CRPS = sigma * (z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi))`. Average the three values within region, then region-weight and normalize exactly as NLL. Use `torch.erf` or `math.erf` through a vectorized NumPy-safe implementation; do not add SciPy.

- [ ] **Step 6: Write and implement coverage/width/Winkler tests**

For interval `[lower, upper]` and miscoverage `alpha`, implement per-reading Winkler score as width plus `2/alpha * (lower-y)` below the interval or `2/alpha * (y-upper)` above it. Average readings within each region before region weighting. Single-reading coverage uses the same region-preserving weighting; simultaneous group coverage is the unweighted fraction of unique groups for which every registered reading of every region is inside.

- [ ] **Step 7: Run probability tests to verify GREEN**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_probability.py -v
```

Expected: all tests pass, including exact boundary inclusion and order-statistic cases.

- [ ] **Step 8: Commit probability primitives**

```powershell
git add src/roughness/sgrpn/probability.py tests/sgrpn/test_probability.py
git commit -m "feat: add group conformal probability metrics"
```

---

### Task 5: Generalize the fixed mean path for three independent seeds

**Files:**
- Modify: `src/roughness/sgrpn/training.py`
- Modify: `tests/sgrpn/test_training.py`

**Interfaces:**
- Consumes: existing Phase A `_fit_p1_subset`, `_fit_r1_nested_stage`, `_fit_g1_nested_stage`, `build_run_fingerprint`, and the fixed data/cache contracts.
- Produces: `MeanPathArtifacts`, `fit_g1_mean_path(config: SGRPNConfig, bundle: DataBundle, cache: OrderSpectrumCache, fold: int, seed: int, train_sample_ids: Sequence[str], *, device, backend=None, batch_size=8) -> MeanPathArtifacts`; existing `run_phase_a_fold` and `run_phase_a` remain behaviorally identical.

- [ ] **Step 1: Write the failing seed-independence test**

```python
def test_phase_b_mean_path_accepts_only_registered_seeds(tmp_path):
    config, bundle, cache = _fixture(tmp_path, max_epochs=2)
    train_index, _ = outer_indices(bundle, fold=0)
    train_sample_ids = bundle.manifest.iloc[train_index]["sample_id"].astype(str).tolist()
    for seed in (20260723, 20260724, 20260725):
        artifacts = fit_g1_mean_path(
            config, bundle, cache, fold=0, seed=seed,
            train_sample_ids=train_sample_ids,
            device="cpu", backend=RecordingBackend(), batch_size=2,
        )
        assert artifacts.seed == seed
    with pytest.raises(ValueError, match="registered Phase B seed"):
        fit_g1_mean_path(
            config, bundle, cache, 0, 7, train_sample_ids,
            device="cpu",
        )
```

- [ ] **Step 2: Write the fixed-stage and fresh-state tests**

Instrument constructors and assert each call executes only `P1`, `R1`, then `G1`; each seed receives fresh parameter objects; G1 experts are frozen; V1/F1 are never created; and no Phase A checkpoint is loaded. Assert the same seed produces byte-identical initial state under the deterministic context seeding function.

- [ ] **Step 3: Run targeted training tests to verify RED**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_training.py -k "phase_b_mean or phase_a_regression" -v
```

Expected: FAIL because `MeanPathArtifacts` and `fit_g1_mean_path` are absent.

- [ ] **Step 4: Add the exact mean-path result**

```python
@dataclass(frozen=True)
class MeanPathArtifacts:
    fold: int
    seed: int
    train_sample_ids: tuple[str, ...]
    model: SelectiveGatedModel
    process_scaler: ProcessScaler
    spectrum_scaler: SpectrumScaler
    quality_scaler: QualityScaler
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]]
    inner_models: Sequence[SelectiveGatedModel]
    inner_scalers: Sequence[tuple[ProcessScaler, SpectrumScaler, QualityScaler]]
    best_epochs: Mapping[str, Sequence[int]]
    histories: Mapping[str, pd.DataFrame]
```

- [ ] **Step 5: Extract the fixed P1→R1→G1 orchestration**

Use the existing fit helpers without changing architectures, loss functions, learning rates, patience, epoch cap, channel swap, scaler boundaries, residual target, or freeze semantics. Normalize `train_sample_ids` to an exact unique ordered subset of the registered outer-train IDs, reject unknown/duplicate/outer-test IDs before any constructor or backend call, and fit every scaler/model only from that subset. The returned inner model at index `k` must have seen only inner-train groups for split `k`; include auditable sample/group tuples for each inner model in checkpoint metadata.

- [ ] **Step 6: Keep Phase A wrappers exact**

Leave `run_phase_a_fold`'s P1/V1/F1/R1/G1 order and seed `20260723` restriction intact. Add regression assertions that its prediction columns, fingerprint, stage order, and persisted marker/state/checkpoint/history/scaler protocol remain `sgrpn-phase-a-v2` and match pre-change fixtures.

- [ ] **Step 7: Run all training tests to verify GREEN**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_training.py -v
```

Expected: all existing Phase A and new mean-path tests pass.

- [ ] **Step 8: Commit the reusable mean path**

```powershell
git add src/roughness/sgrpn/training.py tests/sgrpn/test_training.py
git commit -m "refactor: expose fixed SGRPN mean path"
```

---

### Task 6: Train scale heads with nested group OOF calibration

**Files:**
- Create: `src/roughness/sgrpn/phase_b_training.py`
- Create: `tests/sgrpn/test_phase_b_training.py`

**Interfaces:**
- Consumes: `PhaseBConfig`, `PhaseAHandoff`, `DataBundle`, `OrderSpectrumCache`, `MeanPathArtifacts`, the two scale models, and probability primitives.
- Produces: `ScaleFit`, `CalibrationArtifacts`, `PhaseBFoldArtifacts`, `fit_scale_model`, `build_nested_calibration`, `run_phase_b_fold`, and `run_phase_b`.

- [ ] **Step 1: Write the failing scale-fit freeze test**

```python
def test_scale_fit_cannot_change_mean_model():
    mean = SelectiveGatedModel()
    batch = {
        "spectrum": torch.zeros(2, 1, 3, 361),
        "window_mask": torch.ones(2, 1, dtype=torch.bool),
        "process": torch.zeros(2, 9),
        "quality": torch.zeros(2, 7),
        "readings": torch.tensor([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        "sample_weight": torch.tensor([0.5, 0.5]),
    }
    loader = DataLoader([batch], batch_size=None)
    before = {name: value.clone() for name, value in mean.state_dict().items()}
    fit_scale_model(
        mean_model=mean, scale_model=VarianceHead(), train_loader=loader,
        valid_loader=loader, max_epochs=2, patience=1,
        learning_rate=1e-3, weight_decay=1e-4, device=torch.device("cpu"), seed=20260723,
    )
    assert all(torch.equal(before[name], value) for name, value in mean.state_dict().items())
```

- [ ] **Step 2: Write the calibration-label barrier test**

Use eight groups. Change every reading in one calibration-validation group and assert the recorded train IDs, scaler-source IDs, mean-fit IDs, and scale-fit IDs for that group's predictor are unchanged and exclude that group. Assert the changed labels affect only that group's final nonconformity score, never another group's `mu` or `sigma`.

- [ ] **Step 3: Run Phase B training tests to verify RED**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_phase_b_training.py -v
```

Expected: collection FAIL because `phase_b_training` is absent.

- [ ] **Step 4: Implement exact scale-fit results**

```python
@dataclass(frozen=True)
class ScaleFit:
    model: nn.Module
    best_epoch: int
    history: pd.DataFrame
    mean_state_sha256: str


@dataclass(frozen=True)
class CalibrationArtifacts:
    predictions: pd.DataFrame
    group_scores: pd.DataFrame
    quantiles: Mapping[float, GroupConformalResult]
    inner_fold_definitions: pd.DataFrame


@dataclass(frozen=True)
class PhaseBRunFingerprint:
    value: str
    config_sha256: str
    phase_a_acceptance_sha256: str
    phase_a_run_manifest_sha256: str
    phase_a_training_fingerprint: str
    manifest_sha256: str
    folds_sha256: str
    cache_sha256: str


@dataclass(frozen=True)
class PhaseBFoldArtifacts:
    predictions: pd.DataFrame
    calibration: Mapping[str, CalibrationArtifacts]
    checkpoint_paths: Mapping[str, Path]
    history_paths: Mapping[str, Path]
    fingerprint: PhaseBRunFingerprint


@dataclass(frozen=True)
class PhaseBRunArtifacts:
    probability_predictions: pd.DataFrame
    mean_predictions: pd.DataFrame
    fold_artifacts: Mapping[tuple[int, int], PhaseBFoldArtifacts]
    fingerprints: Mapping[tuple[int, int], PhaseBRunFingerprint]
```

- [ ] **Step 5: Implement `fit_scale_model`**

For each optimizer batch, obtain the frozen mean model's current-orientation output under `torch.no_grad()` and build the fixed 82 features. For validation and all saved inference, call `average_swap_predictions` first and build features from its averaged output, matching the registered mean predictor used for selection and evaluation. Train only scale parameters using `repeated_gaussian_nll`, AdamW with the Phase B configuration, and select the finite minimum validation NLL with patience 20 and maximum 200 epochs. Restore the selected state and call `assert_mean_model_unchanged` before returning.

- [ ] **Step 6: Implement group-confined nested calibration**

For each outer-train calibration split `k`, pass exactly that split's inner-training `sample_id` sequence to `fit_g1_mean_path`, select/refit each scale model using group splits wholly inside that confined training subset, and predict `mu`/`sigma` once for `k`'s unseen groups. Concatenate the four validation blocks, require each outer-train segment exactly once, compute one maximum score per group, then compute 90%/95% order-statistic quantiles. Persist the outer split, every nested split, sample IDs, group IDs, seed, and scaler sources.

- [ ] **Step 7: Add group-score cardinality tests**

Assert `len(group_scores) == outer_train.group_id.nunique()`, `reading_count == 3 * sum(region_count)` per group, no duplicate group, all scores finite/nonnegative, and both quantiles equal an observed score at their registered one-based order index.

- [ ] **Step 8: Fit the final outer-train models only after calibration is complete**

Retrain the fixed G1 mean model on all outer-train groups for the current seed. Select scale epochs with four group folds inside outer train, refit a fresh heteroscedastic head and a fresh global-scale ablation on all outer-train regions, then freeze all models. Do not materialize outer-test readings until mean/scale fitting and both conformal quantiles are finalized.

- [ ] **Step 9: Predict the outer test once and build exact rows**

Use original/swapped spectrum averaging for G1, compute finite positive sigma for both scale models, apply raw and calibrated bounds, then attach outer-test readings solely for scoring. Require the registered column order and one row per `(sample_id, seed, scale_model)`.

- [ ] **Step 10: Add one-fold CPU integration coverage**

With synthetic spectra, eight groups, `max_epochs=2`, and `patience=1`, run one complete outer fold for all three seeds. Assert finite predictions/scales/bounds, positive sigma, no group overlap at any depth, exact per-seed/model coverage, unchanged Phase A fixture hashes, and no output outside the injected Phase B root.

- [ ] **Step 11: Run Phase B training tests to verify GREEN**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_phase_b_training.py -v
```

Expected: all unit and synthetic integration tests pass on CPU.

- [ ] **Step 12: Commit nested probability training**

```powershell
git add src/roughness/sgrpn/phase_b_training.py tests/sgrpn/test_phase_b_training.py
git commit -m "feat: add nested SGRPN probability training"
```

---

### Task 7: Add exact fingerprints, atomic resume, and three-seed outer orchestration

**Files:**
- Modify: `src/roughness/sgrpn/phase_b_training.py`
- Modify: `tests/sgrpn/test_phase_b_training.py`

**Interfaces:**
- Consumes: complete single-fold training from Task 6 and `PhaseAHandoff`.
- Produces: `completed_phase_b_fold_matches`, atomic fold artifacts, `run_phase_b_fold`, and `run_phase_b(...) -> PhaseBRunArtifacts` using the result/fingerprint types defined in Task 6.

- [ ] **Step 1: Write failing fingerprint mismatch tests**

```python
@pytest.mark.parametrize("field", ["fingerprint", "seed", "fold", "models"])
def test_phase_b_resume_rejects_marker_binding_change(completed_fold_fixture, field):
    marker_path, expected_fingerprint = completed_fold_fixture
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload[field] = {"fingerprint": "wrong", "seed": 7, "fold": 9, "models": ["wrong"]}[field]
    marker_path.write_text(json.dumps(payload), encoding="utf-8")
    assert completed_phase_b_fold_matches(
        marker_path, expected_fingerprint, fold=0, seed=20260723
    ) is False
```

The `completed_fold_fixture` must create the exact Task 7 Step 6 tree with valid minimal checkpoint/history/scaler/calibration/prediction bytes, compute every marker hash from those bytes, and return `(marker_path, expected_fingerprint)`. Separate tests alter the Phase B config, Phase A handoff, manifest, folds, cache, and each registered artifact byte; each alteration must change the expected fingerprint or artifact hash and return `False` before reuse.

- [ ] **Step 2: Write the exact completion-artifact test**

Require both scale checkpoints/histories, P1/R1/G1 mean checkpoints/histories, process/spectrum/quality scalers, inner fold definitions, calibration OOF predictions, group scores, quantiles, outer predictions, state JSON, and completion JSON. Delete or corrupt each artifact in turn and require resume rejection before any checkpoint is reused.

- [ ] **Step 3: Run resume tests to verify RED**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_phase_b_training.py -k "fingerprint or resume or completion" -v
```

Expected: FAIL because Phase B completion validation is incomplete.

- [ ] **Step 4: Implement the run fingerprint**

Hash canonical JSON containing all fields plus the exact Phase B seeds, alpha levels, model names, code protocol `sgrpn-phase-b-v1`, fold, and seed. Checkpoint metadata must contain protocol, fold, seed, model, best epochs, refit epoch, fingerprint, and mean-state hash.

- [ ] **Step 5: Implement atomic stage publication**

Write bytes, JSON, CSV, NPZ, and PyTorch checkpoints through unique sibling temporary files followed by `Path.replace`. Publish `complete.json` last, only after reloading and deeply validating every artifact and recomputing every registered hash.

- [ ] **Step 6: Implement the exact directory state machine**

```text
outputs/sgrpn/phase_b/folds/fold_<k>/seed_<seed>/
  mean/checkpoints/{P1,R1,G1}.pt
  mean/history/{P1,R1,G1}.csv
  scale/checkpoints/{heteroscedastic,homoscedastic}.pt
  scale/history/{heteroscedastic,homoscedastic}.csv
  scalers/{process,spectrum,quality}.npz
  calibration/inner_folds.csv
  calibration/oof_predictions.csv
  calibration/group_scores.csv
  calibration/quantiles.json
  predictions.csv
  state.json
  complete.json
```

State order is `mean`, `heteroscedastic`, `homoscedastic`, `calibration`, `prediction`, `complete`. Resume only an exact completed seed/fold; reject partial directories whose last durable state or hash does not match.

- [ ] **Step 7: Implement all-fold/all-seed orchestration**

`run_phase_b` must iterate the registered five folds and three seeds without reading metrics between runs, concatenate 3516 probability rows and 5274 P1/R1/G1 mean rows, validate both exact Cartesian products, atomically write `outputs/sgrpn/phase_b/predictions/oof_probability_predictions.csv` and `outputs/sgrpn/phase_b/predictions/oof_mean_predictions.csv`, and return a `PhaseBRunArtifacts` whose `fold_artifacts` and `fingerprints` mappings both have exactly the 15 registered `(fold, seed)` keys.

- [ ] **Step 8: Run resume and orchestration tests to verify GREEN**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_phase_b_training.py -v
```

Expected: all tests pass; interrupted fixture resumes only exact completed units.

- [ ] **Step 9: Commit Phase B orchestration**

```powershell
git add src/roughness/sgrpn/phase_b_training.py tests/sgrpn/test_phase_b_training.py
git commit -m "feat: orchestrate SGRPN phase B folds"
```

---

### Task 8: Evaluate probability quality and enforce claim limits

**Files:**
- Modify: `src/roughness/sgrpn/evaluation.py`
- Modify: `tests/sgrpn/test_evaluation.py`

**Interfaces:**
- Consumes: exact Phase B OOF rows, saved calibration score/quantile tables, and registered manifest/folds.
- Produces: `ProbabilityMetrics`, `PhaseBPaperDecision`, `validate_phase_b_prediction_cartesian`, `probability_metric_table`, `seed_uncertainty_summary`, `paired_probability_bootstrap`, and `assess_phase_b_claims`.

- [ ] **Step 1: Write the failing Cartesian validation tests**

Construct 586 fixture IDs with three seeds and two scale models. Assert valid rows pass; remove one pair, duplicate one pair, swap one group, change one reading, use a noncanonical seed, use sigma zero, or change one conformal quantile within a fold and assert each case fails.

- [ ] **Step 2: Write exact metric-table tests**

```python
def test_probability_table_contains_every_registered_measure():
    rows = []
    for group_id, mu in (("g1", 0.5), ("g2", 0.8)):
        for scale_model in SCALE_MODELS:
            rows.append({
                "sample_id": group_id, "group_id": group_id, "version": "v3",
                "fold": 0, "seed": 20260723, "scale_model": scale_model,
                "target_mean": mu, "ra_1": mu - 0.1, "ra_2": mu,
                "ra_3": mu + 0.1, "sample_weight": 1.0, "mu": mu,
                "sigma": 0.2, "gate": 0.5, "correction": 0.0,
                "raw_lower_90": mu - 0.4, "raw_upper_90": mu + 0.4,
                "raw_lower_95": mu - 0.5, "raw_upper_95": mu + 0.5,
                "conformal_q_90": 2.0, "conformal_lower_90": mu - 0.4,
                "conformal_upper_90": mu + 0.4, "conformal_q_95": 2.5,
                "conformal_lower_95": mu - 0.5, "conformal_upper_95": mu + 0.5,
            })
    table = probability_metric_table(pd.DataFrame(rows, columns=PHASE_B_PREDICTION_COLUMNS))
    assert set(table["metric"]) == {
        "mean_mae", "mean_rmse", "mean_r2", "gaussian_nll", "gaussian_crps",
        "single_reading_coverage", "simultaneous_group_coverage",
        "mean_interval_width", "winkler_score",
    }
    assert set(table["interval_type"].dropna()) == {"raw", "conformal"}
    assert set(table["nominal_coverage"].dropna()) == {0.90, 0.95}
```

- [ ] **Step 3: Run evaluation tests to verify RED**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_evaluation.py -k phase_b -v
```

Expected: FAIL because Phase B evaluation interfaces are absent.

- [ ] **Step 4: Implement exact OOF and metric validation**

Validate 3516 probability rows and 5274 mean rows, all metadata invariants, `target_mean == mean(ra_1,ra_2,ra_3)`, `sample_weight == 1/split_count` after joining manifest, fold equality with registered folds, positive finite sigma, interval ordering, and exact alpha/quantile semantics. Calculate metrics per `(seed, scale_model, interval_type, nominal_coverage)`, fold-level rows, and all-seed summaries that retain seed as the replication unit. From the mean table, calculate weighted P1/R1/G1 MAE/RMSE/R², raw and `>0.01 µm` material negative-transfer rates relative to P1, 10,000 group-paired Bootstrap differences, and gate distributions overall and by `n_rpm`, `fz_mm_per_tooth`, `ap_mm`, and all seven quality features.

- [ ] **Step 5: Implement descriptive seed uncertainty without new intervals**

For each `(sample_id, scale_model)`, save `mu_mean`, `aleatoric_variance_mean = mean(sigma_seed²)`, and `seed_prediction_variance = mean((mu_seed-mu_mean)²)`. Label the last field `descriptive_seed_instability`; do not call it calibrated epistemic uncertainty, do not form `total_variance`, and do not evaluate ensemble coverage.

- [ ] **Step 6: Implement group-paired bootstrap**

For heteroscedastic minus homoscedastic NLL, CRPS, width, and Winkler comparisons, resample whole `group_id` blocks exactly 10,000 times with seed `20260723`; preserve all three seed rows inside a sampled group. Store point estimate, 95% percentile interval, repetitions, and `resampling_unit="group_id"`.

- [ ] **Step 7: Write and implement the paper-decision tests**

```python
@dataclass(frozen=True)
class ProbabilityMetrics:
    mean_mae: float
    mean_rmse: float
    mean_r2: float
    gaussian_nll: float
    gaussian_crps: float
    single_reading_coverage: float
    simultaneous_group_coverage: float
    mean_interval_width: float
    winkler_score: float


@dataclass(frozen=True)
class PhaseBPaperDecision:
    emphasize_heteroscedasticity: bool
    retain_group_conformal: bool
    practical_width_threshold_registered: bool
    coverage_claim_scope: str
    sigma_interpretation: str
    seed_spread_interpretation: str
    reasons: Sequence[str]


def test_no_winkler_gain_disables_heteroscedastic_emphasis():
    metrics = pd.DataFrame([
        {"aggregation": "all_seed", "scale_model": "heteroscedastic", "interval_type": "conformal", "nominal_coverage": 0.90, "metric": "winkler_score", "value": 1.01},
        {"aggregation": "all_seed", "scale_model": "heteroscedastic", "interval_type": "conformal", "nominal_coverage": 0.95, "metric": "winkler_score", "value": 1.01},
        {"aggregation": "all_seed", "scale_model": "homoscedastic", "interval_type": "conformal", "nominal_coverage": 0.90, "metric": "winkler_score", "value": 1.00},
        {"aggregation": "all_seed", "scale_model": "homoscedastic", "interval_type": "conformal", "nominal_coverage": 0.95, "metric": "winkler_score", "value": 1.00},
    ])
    decision = assess_phase_b_claims(metrics)
    assert decision.emphasize_heteroscedasticity is False
    assert decision.retain_group_conformal is True
    assert decision.practical_width_threshold_registered is False
```

Set `emphasize_heteroscedasticity=true` only when the all-seed conformal Winkler score is finite and strictly lower for heteroscedastic than homoscedastic at both 90% and 95%. Always retain and report group conformal calibration. The reasons must state when the score rule fails and that no numeric practical-width threshold was preregistered.

- [ ] **Step 8: Run evaluation tests to verify GREEN**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_evaluation.py -v
```

Expected: all Phase A and Phase B evaluation tests pass.

- [ ] **Step 9: Commit probability evaluation**

```powershell
git add src/roughness/sgrpn/evaluation.py tests/sgrpn/test_evaluation.py
git commit -m "feat: evaluate SGRPN probability intervals"
```

---

### Task 9: Publish reproducible Phase B outputs and CLI commands

**Files:**
- Modify: `src/roughness/sgrpn/reporting.py`
- Modify: `src/roughness/sgrpn/cli.py`
- Modify: `tests/sgrpn/test_reporting.py`
- Modify: `tests/sgrpn/test_cli.py`

**Interfaces:**
- Consumes: `PhaseBConfig`, validated handoff, complete Phase B OOF rows, calibration artifacts, metric tables, bootstraps, and paper decision.
- Produces: `write_phase_b_report` with the registered signature, `validate_phase_b_outputs(config: PhaseBConfig) -> None`, and four CLI commands.

- [ ] **Step 1: Write failing report reconstruction tests**

Generate a synthetic validated OOF CSV and JSON quantiles, call `write_phase_b_report`, delete only generated figures/tables, regenerate them without checkpoints, and assert identical table bytes and present figures. Hash Phase A and both legacy roots before/after and require equality.

- [ ] **Step 2: Write failing CLI gate/order tests**

Assert `train-phase-b`, `evaluate-phase-b`, and `run-phase-b` call `validate_phase_b_handoff` before creating the output root. On a fresh run, assert `run-phase-b` orders `preflight → train → evaluate → validate`; on a fully completed and evaluated `--resume` run, assert it skips training/evaluation and performs deep validation only. A closed/tampered gate returns nonzero and invokes no trainer or writer.

- [ ] **Step 3: Run reporting/CLI tests to verify RED**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_reporting.py tests/sgrpn/test_cli.py -k phase_b -v
```

Expected: FAIL because Phase B reporting and commands are absent.

- [ ] **Step 4: Implement the exact report tree**

```text
outputs/sgrpn/phase_b/
  handoff.json
  run_manifest.json
  immutable_hashes_before.json
  immutable_hashes_after.json
  folds/fold_<k>/seed_<seed>/**
  folds/outer_folds.csv
  folds/inner_folds.csv
  predictions/oof_probability_predictions.csv
  predictions/oof_mean_predictions.csv
  predictions/group_oof_predictions.csv
  predictions/seed_uncertainty_summary.csv
  calibration/group_scores.csv
  calibration/quantiles.csv
  evaluation/mean_metrics.csv
  evaluation/probability_metrics.csv
  evaluation/fold_metrics.csv
  evaluation/seed_metrics.csv
  evaluation/ablation_differences.csv
  evaluation/paired_bootstrap.csv
  evaluation/negative_transfer.csv
  evaluation/gate_statistics.csv
  evaluation/claim_decision.json
  evaluation/method_notes.json
  evaluation/breakdowns/by_version.csv
  evaluation/breakdowns/by_n_rpm.csv
  evaluation/breakdowns/by_fz_mm_per_tooth.csv
  evaluation/breakdowns/by_ap_mm.csv
  evaluation/figures/coverage_width.png
  evaluation/figures/interval_score.png
  evaluation/figures/nll_crps_by_seed.png
  evaluation/figures/group_simultaneous_coverage.png
  evaluation/figures/prediction_scatter.png
  evaluation/figures/residual_plot.png
  evaluation/figures/fold_stability.png
  evaluation/figures/gate_distribution.png
  evaluation/figures/gate_condition_heatmap.png
```

- [ ] **Step 5: Write exact method and claim notes**

`method_notes.json` must state: three readings remain one region; probability loss preserves region weight; `sigma` combines repeat dispersion and unmodeled error; no Gauge R&R or variance-source decomposition; seed spread is descriptive instability only; `version` is excluded from inputs; v3/v4 is a composite-domain stress test confounded with speed; Ch9/Ch10 orientation is unresolved; coverage applies only to exchangeable new groups of existing type; no numeric practical-width threshold was registered; no outer result changed the design.

- [ ] **Step 6: Bind every report to saved inputs**

`run_manifest.json` must include Phase B config/hash, Phase A handoff hashes, exact seeds/device/environment, input/cache/fold fingerprints, code protocol, each fold completion hash, output artifact hashes, bootstrap definition, and selected device per fold/seed. `handoff.json` is the serialized `PhaseAHandoff`; it must not rewrite Phase A's manifest. The breakdown and figure writers must include the v3/v4 composite-domain result with its speed-confounding statement and regenerate the mean prediction, residual, fold-stability, gate-distribution, gate-condition, coverage-width, interval-score, NLL/CRPS, and simultaneous-coverage figures only from persisted tables.

- [ ] **Step 7: Implement output integrity checks**

`validate_phase_b_outputs` must reload all CSV/JSON/checkpoints, re-run Cartesian, group-score, quantile, interval, metric, bootstrap, and fingerprint checks, require all 15 `(fold,seed)` completions, and compare `immutable_hashes_before.json` with freshly computed Phase A/legacy hashes byte-for-byte.

- [ ] **Step 8: Add the exact CLI surface**

```text
roughness-sgrpn preflight-phase-b --config configs/sgrpn_phase_b.yaml
roughness-sgrpn train-phase-b --config configs/sgrpn_phase_b.yaml [--fold 0] [--seed 20260723] [--device auto] [--resume]
roughness-sgrpn evaluate-phase-b --config configs/sgrpn_phase_b.yaml
roughness-sgrpn run-phase-b --config configs/sgrpn_phase_b.yaml [--device auto] [--resume]
```

`--fold` accepts only 0–4; `--seed` accepts only the three registered seeds. `--device auto` records CUDA or CPU. Return nonzero on a gate/fingerprint mismatch, leakage assertion, incomplete OOF/calibration coverage, nonfinite value, invalid scale/interval, stale checkpoint, output-root escape, or immutable-input hash change.

- [ ] **Step 9: Run reporting and CLI tests to verify GREEN**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_reporting.py tests/sgrpn/test_cli.py -v
```

Expected: all Phase A and Phase B reporting/CLI tests pass.

- [ ] **Step 10: Commit reporting and CLI support**

```powershell
git add src/roughness/sgrpn/reporting.py src/roughness/sgrpn/cli.py tests/sgrpn/test_reporting.py tests/sgrpn/test_cli.py
git commit -m "feat: report SGRPN phase B results"
```

---

### Task 10: Verify the complete implementation before formal execution

**Files:**
- Modify only if a test exposes a defect: files already listed in Tasks 1–9
- Create through verified test evidence: `outputs/sgrpn/phase_b/test_evidence.json`

**Interfaces:**
- Consumes: the complete Phase B implementation and synthetic fixtures.
- Produces: passing compile, focused, SGRPN, and full-regression evidence; it does not run formal Phase B training.

- [ ] **Step 1: Compile the package and tests**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m compileall -q src/roughness/sgrpn tests/sgrpn
```

Expected: exit 0.

- [ ] **Step 2: Run focused Phase B tests**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_config.py tests/sgrpn/test_models.py tests/sgrpn/test_probability.py tests/sgrpn/test_phase_b_training.py tests/sgrpn/test_evaluation.py tests/sgrpn/test_reporting.py tests/sgrpn/test_cli.py -v
```

Expected: zero failures.

- [ ] **Step 3: Run the complete SGRPN suite**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn -q
```

Expected: zero failures, including all Phase A regression tests.

- [ ] **Step 4: Run the entire project regression suite**

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest -q
```

Expected: zero failures.

- [ ] **Step 5: Exercise only the read-only formal preflight**

```powershell
E:\CodeX\机床项目\.venv\Scripts\roughness-sgrpn.exe preflight-phase-b --config configs/sgrpn_phase_b.yaml
```

Expected: exit 0 with Phase A acceptance path `transfer_safety`, matching hashes/fingerprints, five complete folds, 212 groups, and 586 segments. This command must not create `outputs/sgrpn/phase_b/`.

- [ ] **Step 6: Record test evidence atomically**

Write command strings, exit codes, pass/fail counts, elapsed seconds, Python executable/version, package versions, and the verified Phase A handoff hash to `outputs/sgrpn/phase_b/test_evidence.json`. Do not record a formal-training status.

- [ ] **Step 7: Commit verification fixes and compact evidence**

If Steps 1–5 required corrections, commit only the corrected source/tests and compact JSON evidence:

```powershell
git add src/roughness/sgrpn tests/sgrpn outputs/sgrpn/phase_b/test_evidence.json
git commit -m "test: verify SGRPN phase B pipeline"
```

Do not add checkpoints, formal predictions, or large caches.

---

### Task 11: Execute and verify the formal Phase B experiment

**Files:**
- Create through the tested CLI: `outputs/sgrpn/phase_b/**`
- Verify read-only: `configs/sgrpn_phase_a.yaml`
- Verify read-only: `outputs/sgrpn/phase_a/**`
- Verify read-only: `outputs/scheme1/**`
- Verify read-only: `outputs/scheme1_physics/**`

**Interfaces:**
- Consumes: the fully tested Phase B CLI, immutable validated Phase A handoff, frozen registered inputs, and no additional experiment.
- Produces: complete five-fold × three-seed × two-scale-model OOF probability evidence, group-conformal intervals, ablation results, and the machine-readable paper-claim decision.

- [ ] **Step 1: Re-run preflight and capture immutable hashes**

```powershell
E:\CodeX\机床项目\.venv\Scripts\roughness-sgrpn.exe preflight-phase-b --config configs/sgrpn_phase_b.yaml
```

Then use the reporting hash helper to atomically write every file hash under the Phase A and both legacy roots to `outputs/sgrpn/phase_b/immutable_hashes_before.json`. Resolve junctions and require the exact intended roots before enumerating files.

- [ ] **Step 2: Run all formal Phase B training without intermediate metric inspection**

```powershell
E:\CodeX\机床项目\.venv\Scripts\roughness-sgrpn.exe train-phase-b --config configs/sgrpn_phase_b.yaml --device auto --resume
```

Expected: 15 exact `(fold,seed)` completion markers; each contains P1/R1/G1, both scale models, four inner calibration folds, 90%/95% quantiles, finite outer predictions, the selected device, and the Phase B/Phase A fingerprints. Monitor only state/completion fields, not metrics, predictions, scales, intervals, or comparisons.

- [ ] **Step 3: Evaluate exactly once after all training is complete**

```powershell
E:\CodeX\机床项目\.venv\Scripts\roughness-sgrpn.exe evaluate-phase-b --config configs/sgrpn_phase_b.yaml
```

Expected: exit 0 and all registered region/group prediction, calibration, mean/probability metric, Bootstrap, negative-transfer, gate, ablation, composite-domain, plot, method-note, and claim-decision outputs. Do not alter or rerun the experiment based on observed seed/fold results.

- [ ] **Step 4: Verify artifacts and immutable inputs**

Recompute hashes into `immutable_hashes_after.json`, require exact equality with `immutable_hashes_before.json`, then run:

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -c "from roughness.sgrpn.config import load_phase_b_config; from roughness.sgrpn.reporting import validate_phase_b_outputs; validate_phase_b_outputs(load_phase_b_config('configs/sgrpn_phase_b.yaml'))"
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn -q
```

Expected: the read-only validator does not retrain, reevaluate, or rewrite predictions; tests have zero failures. Require 3516 exact probability OOF rows, 5274 exact mean OOF rows, 15 completions, positive finite scales, valid intervals, one calibration score per group per fold/seed/model, observed-order-statistic quantiles, group bootstrap unit, and exactly 10,000 repetitions.

- [ ] **Step 5: Apply the registered paper stopping rule without reinterpretation**

Read `outputs/sgrpn/phase_b/evaluation/claim_decision.json` once after all independent checks. Report `emphasize_heteroscedasticity`, `retain_group_conformal`, exact reasons, both coverage levels, widths, Winkler scores, NLL, CRPS, and seed stability. If heteroscedasticity is not emphasized, retain the ablation and failures and limit the paper to group conformal calibration plus the registered caveats. Never tune, add an experiment, drop a seed/fold, or invent a practical-width threshold.

- [ ] **Step 6: Preserve formal evidence without committing large artifacts**

Do not commit binary checkpoints, NPZ files, or formal prediction matrices unless the user explicitly requests them. If repository policy permits compact experiment evidence, stage only configuration, source, tests, compact CSV/JSON summaries, and requested figures after the user reviews the formal outcome.

---

## Execution Checkpoints

1. Before Task 1 edits: Phase A gate, acceptance hash, training fingerprint, cache/input hashes, and five completion markers all match; otherwise stop with no Phase B write.
2. After Task 3: prove positive finite scale output, exact 82-feature ordering, region-preserving repeated NLL, and unchanged mean-model tensors.
3. After Task 6: inspect one synthetic nested calibration audit and prove each validation group is absent from every fit/selection/scaler source used for its `mu` and `sigma`.
4. After Task 7: corrupt one artifact of every type and prove resume rejects it before model reuse.
5. After Task 9: regenerate all tables/figures from saved CSV/JSON inputs and prove Phase A/legacy byte hashes are unchanged.
6. After Task 10: require focused, SGRPN, and full project suites to pass before formal training.
7. After Task 11: report the emitted claim decision exactly, preserve unfavorable results, and stop without result-driven modification.
