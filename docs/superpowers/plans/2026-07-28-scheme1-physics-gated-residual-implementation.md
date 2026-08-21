# 方案1物理补充分支实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改原方案1正式产物的前提下，实现PW0/PW1/PW2与PE0/PE1/PE2双公式物理基线、普通残差和两阶段门控残差的完整训练、评估与报告流程。

**Architecture:** 新模块只读复用方案1的manifest、固定外层折、窗口索引、逐折通道统计、`CNNEncoder`和M0 OOF结果。每个“公式 × 外层折 × 种子”只在外层训练数据内选择一次有效尺度，随后由物理单独预测、普通残差和门控残差共享；门控模型加载并冻结普通残差模型，只训练基于四维工况输入的Logistic门控。

**Tech Stack:** Python 3.12、PyTorch、pandas、NumPy、scikit-learn、Matplotlib、PyYAML、pytest。

## Global Constraints

- 设计规格：`docs/superpowers/specs/2026-07-28-scheme1-physics-gated-residual-design.md`。
- 原方案1的 `configs/scheme1.yaml` 与 `outputs/scheme1/` 只读，不得覆盖。
- 新代码、配置、测试和输出分别位于 `src/roughness/scheme1_physics/`、`configs/scheme1_physics.yaml`、`tests/scheme1_physics/`、`outputs/scheme1_physics/`。
- 模型集合固定为 `PW0,PW1,PW2,PE0,PE1,PE2`；M0只作外部参照。
- 半径候选固定为 `[0.075, 0.10, 0.20, 0.40, 0.80]` mm。
- 外层5折按 `group_id` 分组；随机种子固定为 `20260723,20260724,20260725`。
- 最大200轮、patience 20、AdamW、学习率0.001、权重衰减0.0001、batch size 4、加权MAE损失。
- 所有标准化、尺度选择、早停和checkpoint选择仅使用当前外层训练数据。
- 门控输入固定为 `[n_rpm,fz_mm_per_tooth,ap_mm,Ra_physical]`，CNN嵌入不得进入门控。
- 门控预测恒等式固定为 `physical + (1-gate) * residual`。
- 当前工作区不是Git仓库。不得擅自初始化Git；每个任务末尾列出的提交命令仅在用户以后将目录置于Git仓库时执行。
- 项目验证命令统一使用 `.\.venv\Scripts\python.exe`；该虚拟环境基于用户指定的 `D:\CodexPython\python.exe`。

---

## 文件结构与职责

| 文件 | 职责 |
|---|---|
| `src/roughness/scheme1_physics/config.py` | 加载补充分支配置并验证源方案1产物 |
| `src/roughness/scheme1_physics/physics.py` | 双公式、候选尺度选择、物理OOF与选择溯源 |
| `src/roughness/scheme1_physics/data.py` | 外层折数据准备、训练折标准化、Dataset/DataLoader构造 |
| `src/roughness/scheme1_physics/models.py` | CNN＋64/32 MLP残差与Logistic门控 |
| `src/roughness/scheme1_physics/training.py` | 普通残差训练、冻结门控训练、fingerprint与OOF落盘 |
| `src/roughness/scheme1_physics/evaluation.py` | 指标、消融门槛、公式比较、Bootstrap与最终分级 |
| `src/roughness/scheme1_physics/reporting.py` | 预测、残差、工况误差、公式差异和门控分布图 |
| `src/roughness/scheme1_physics/cli.py` | `prepare/baselines/train/evaluate/run` 编排与断点续跑 |
| `configs/scheme1_physics.yaml` | 源方案1配置、独立输出目录与固定实验参数 |
| `tests/scheme1_physics/` | 与上述模块一一对应的单元和集成测试 |

### 共享测试夹具约定

`tests/scheme1_physics/conftest.py`必须提供一个可重复的微型工作区，避免各测试引用未定义夹具：

```python
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@dataclass(frozen=True)
class TinyWorkspace:
    source_config_path: Path
    physics_config_path: Path
    manifest: pd.DataFrame
    folds: pd.DataFrame
    source_output_dir: Path
    physics_output_dir: Path


@pytest.fixture
def tiny_workspace(tmp_path: Path) -> TinyWorkspace:
    segments = tmp_path / "segments"
    source_output = tmp_path / "scheme1"
    physics_output = tmp_path / "scheme1_physics"
    segments.mkdir()
    (source_output / "folds").mkdir(parents=True)
    (source_output / "classic").mkdir(parents=True)
    rows = []
    fold_rows = []
    for index in range(10):
        sample_id = f"s{index}"
        group_id = f"g{index}"
        signal_path = segments / f"{sample_id}.csv"
        time = np.arange(256, dtype=float)
        pd.DataFrame(
            {
                "Ch9_g": np.sin(time / 11.0 + index),
                "Ch10_g": np.cos(time / 13.0 + index),
                "Ch11_g": np.sin(time / 17.0 + index),
            }
        ).to_csv(signal_path, index=False)
        rows.append(
            {
                "sample_id": sample_id,
                "group_id": group_id,
                "signal_path": str(signal_path),
                "n_rpm": 6000.0 + 500.0 * (index % 3),
                "fz_mm_per_tooth": 0.02 + 0.01 * (index % 4),
                "ap_mm": 0.5 + 0.5 * (index % 2),
                "ra_mean": 0.2 + 0.03 * index,
                "sample_weight": 1.0,
                "version": "tiny",
            }
        )
        fold_rows.append({"sample_id": sample_id, "fold": index % 5})
    manifest = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows)
    manifest_path = tmp_path / "manifest.csv"
    folds_path = tmp_path / "folds.csv"
    manifest.to_csv(manifest_path, index=False)
    folds.to_csv(folds_path, index=False)
    pd.DataFrame(
        {
            "segment_id": manifest["sample_id"],
            "window_id": 0,
            "start_sample": 0,
            "end_sample": 256,
        }
    ).to_csv(source_output / "window_index.csv", index=False)
    for fold in range(5):
        train_ids = manifest.loc[
            folds["fold"].to_numpy() != fold, "sample_id"
        ].astype(str)
        (source_output / "folds" / f"fold_{fold}_channel_stats.json").write_text(
            json.dumps(
                {
                    "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
                    "sample_count": int(len(train_ids) * 256),
                    "source_segment_ids": train_ids.tolist(),
                }
            ),
            encoding="utf-8",
        )
    m0_rows = []
    for seed in (20260723, 20260724, 20260725):
        for row, fold_row in zip(rows, fold_rows, strict=True):
            m0_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "model": "M0",
                    "seed": seed,
                    "fold": fold_row["fold"],
                    "y_true": row["ra_mean"],
                    "y_pred": row["ra_mean"] + 0.02,
                    "sample_weight": 1.0,
                }
            )
    pd.DataFrame(m0_rows).to_csv(
        source_output / "classic" / "oof_predictions.csv", index=False
    )
    source_config = tmp_path / "scheme1.yaml"
    source_config.write_text(
        "\n".join(
            [
                f"segments_dir: {segments.as_posix()}",
                f"manifest_path: {manifest_path.as_posix()}",
                f"folds_path: {folds_path.as_posix()}",
                f"output_dir: {source_output.as_posix()}",
                "sample_rate_hz: 256",
                "window_samples: 256",
                "stride_samples: 256",
                "re_candidates_mm: [0.075, 0.10, 0.20, 0.40, 0.80]",
                "seeds: [20260723, 20260724, 20260725]",
                "inner_splits: 2",
                "welch_nperseg: 128",
                "welch_noverlap: 64",
                "nominal_band_min_halfwidth_hz: 5.0",
                "nominal_band_relative_halfwidth: 0.05",
                "hf_band_hz: [10.0, 100.0]",
                "max_epochs: 2",
                "patience: 1",
                "learning_rate: 0.001",
                "weight_decay: 0.0001",
                "bootstrap_repetitions: 100",
            ]
        ),
        encoding="utf-8",
    )
    physics_config = tmp_path / "scheme1_physics.yaml"
    physics_config.write_text(
        "\n".join(
            [
                f"scheme1_config_path: {source_config.as_posix()}",
                f"output_dir: {physics_output.as_posix()}",
                "models: [PW0, PW1, PW2, PE0, PE1, PE2]",
                "batch_size: 2",
                "validation_fraction: 0.25",
                "residual_dropout: 0.0",
            ]
        ),
        encoding="utf-8",
    )
    return TinyWorkspace(
        source_config,
        physics_config,
        manifest,
        folds,
        source_output,
        physics_output,
    )
```

在同一文件中提供以下薄夹具，所有后续测试只能使用这些名称或在本测试文件内明确构造数据：

```python
@pytest.fixture
def miniature_manifest(tiny_workspace):
    return tiny_workspace.manifest.copy()


@pytest.fixture
def fixed_folds(tiny_workspace):
    return tiny_workspace.folds.copy()


@pytest.fixture
def valid_scheme1_config(tiny_workspace):
    return lambda: tiny_workspace.source_config_path


@pytest.fixture
def valid_physics_config(tiny_workspace):
    def factory(source_equals_destination=False):
        if not source_equals_destination:
            return tiny_workspace.physics_config_path
        text = tiny_workspace.physics_config_path.read_text(encoding="utf-8")
        text = text.replace(
            tiny_workspace.physics_output_dir.as_posix(),
            tiny_workspace.source_output_dir.as_posix(),
        )
        path = tiny_workspace.physics_config_path.with_name("same_output.yaml")
        path.write_text(text, encoding="utf-8")
        return path
    return factory
```

Task 3加入 `train_frame`、`outer_test_frame`、`physics_config`和 `prepared_fold`；Task 5加入 `tiny_physics_config`作为 `physics_config` 的别名。Task 6的 `pe1_checkpoint`、`trained_pw2_result`必须由同一测试文件中的1轮微型训练工厂生成。Tasks 7-8的 `complete_oof`、`comparison_metrics`、`formula_predictions`和 `gated_oof`必须在各自测试文件中用固定数值显式构造，不允许依赖正式输出。

---

### Task 1：建立隔离配置与协议检查点

**Files:**
- Create: `src/roughness/scheme1_physics/__init__.py`
- Create: `src/roughness/scheme1_physics/config.py`
- Create: `configs/scheme1_physics.yaml`
- Create: `tests/scheme1_physics/__init__.py`
- Create: `tests/scheme1_physics/conftest.py`
- Create: `tests/scheme1_physics/test_config.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `roughness.scheme1.config.Scheme1Config` 与 `load_scheme1_config(path)`。
- Produces:
  - `Scheme1PhysicsConfig`
  - `load_scheme1_physics_config(path: str | Path) -> Scheme1PhysicsConfig`
  - `write_physics_protocol_checkpoint(config, output_path=None) -> Path`

- [ ] **Step 1：写配置失败测试**

```python
from pathlib import Path

import pytest

from roughness.scheme1_physics.config import load_scheme1_physics_config


def test_config_keeps_source_and_destination_separate(
    tmp_path: Path, valid_scheme1_config
):
    source = valid_scheme1_config()
    config = tmp_path / "physics.yaml"
    config.write_text(
        "\n".join(
            [
                f"scheme1_config_path: {source.as_posix()}",
                f"output_dir: {(tmp_path / 'physics').as_posix()}",
                "models: [PW0, PW1, PW2, PE0, PE1, PE2]",
                "batch_size: 4",
                "validation_fraction: 0.2",
                "residual_dropout: 0.3",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source Scheme 1 inputs"):
        load_scheme1_physics_config(config)


def test_config_rejects_source_output_as_destination(valid_physics_config):
    path = valid_physics_config(source_equals_destination=True)
    with pytest.raises(ValueError, match="must differ"):
        load_scheme1_physics_config(path)
```

- [ ] **Step 2：运行测试并确认因导入失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1_physics\test_config.py -v
```

Expected: FAIL，提示 `roughness.scheme1_physics.config` 不存在。

- [ ] **Step 3：实现配置对象和验证**

```python
from dataclasses import dataclass
import json
from pathlib import Path

import yaml

from roughness.scheme1.config import Scheme1Config, load_scheme1_config


@dataclass(frozen=True)
class Scheme1PhysicsConfig:
    source: Scheme1Config
    output_dir: Path
    models: tuple[str, ...]
    batch_size: int
    validation_fraction: float
    residual_dropout: float


EXPECTED_MODELS = ("PW0", "PW1", "PW2", "PE0", "PE1", "PE2")


def load_scheme1_physics_config(path: str | Path) -> Scheme1PhysicsConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required = {
        "scheme1_config_path",
        "output_dir",
        "models",
        "batch_size",
        "validation_fraction",
        "residual_dropout",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    source_path = (config_path.parent / raw["scheme1_config_path"]).resolve()
    source = load_scheme1_config(source_path)
    output_dir = (config_path.parent / raw["output_dir"]).resolve()
    if output_dir == source.output_dir:
        raise ValueError("physics output_dir must differ from source output_dir")
    models = tuple(map(str, raw["models"]))
    if models != EXPECTED_MODELS:
        raise ValueError(f"models must equal {EXPECTED_MODELS}")
    validation_fraction = float(raw["validation_fraction"])
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    return Scheme1PhysicsConfig(
        source=source,
        output_dir=output_dir,
        models=models,
        batch_size=int(raw["batch_size"]),
        validation_fraction=validation_fraction,
        residual_dropout=float(raw["residual_dropout"]),
    )
```

`load_scheme1_physics_config`还必须验证以下源文件存在：

```python
required_source_artifacts = [
    source.output_dir / "window_index.csv",
    source.output_dir / "classic" / "oof_predictions.csv",
    *[
        source.output_dir / "folds" / f"fold_{fold}_channel_stats.json"
        for fold in range(5)
    ],
]
```

缺少任一文件时抛出包含 `source Scheme 1 inputs` 的 `ValueError`。

- [ ] **Step 4：写入固定配置**

```yaml
scheme1_config_path: scheme1.yaml
output_dir: ../outputs/scheme1_physics
models: [PW0, PW1, PW2, PE0, PE1, PE2]
batch_size: 4
validation_fraction: 0.2
residual_dropout: 0.3
```

- [ ] **Step 5：注册独立CLI**

在 `pyproject.toml` 的 `[project.scripts]` 增加：

```toml
roughness-scheme1-physics = "roughness.scheme1_physics.cli:main"
```

- [ ] **Step 6：实现协议检查点**

`write_physics_protocol_checkpoint`写入 `outputs/scheme1_physics/run_manifest.json`，至少包含源配置绝对路径、源输出绝对路径、新输出绝对路径、候选尺度、种子、模型集合、训练超参数和 `git_available: false`。

- [ ] **Step 7：运行配置测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1_physics\test_config.py -v
```

Expected: PASS。

- [ ] **Step 8：任务检查点**

当前不执行Git命令。若以后目录成为Git仓库，建议提交：

```powershell
git add pyproject.toml configs/scheme1_physics.yaml src/roughness/scheme1_physics tests/scheme1_physics/test_config.py
git commit -m "feat: scaffold isolated scheme1 physics branch"
```

---

### Task 2：实现双公式和无泄漏尺度选择

**Files:**
- Create: `src/roughness/scheme1_physics/physics.py`
- Create: `tests/scheme1_physics/test_physics.py`

**Interfaces:**
- Consumes:
  - `make_group_inner_splits(frame, n_splits, seed)`
  - manifest列 `sample_id,group_id,fz_mm_per_tooth,ra_mean,sample_weight`
- Produces:
  - `word_ra_um(fz_mm, radius_mm) -> float | np.ndarray`
  - `exact_ra_um(fz_mm, radius_mm) -> float | np.ndarray`
  - `physical_ra_um(formula, fz_mm, radius_mm) -> float | np.ndarray`
  - `ScaleSelection`
  - `select_effective_scale(train_frame, formula, candidates_mm, inner_splits, outer_fold, seed) -> tuple[ScaleSelection, pd.DataFrame]`
  - `build_physics_outer_frames(manifest, folds, outer_fold, seed, formula, candidates_mm, inner_splits) -> tuple[pd.DataFrame, pd.DataFrame, ScaleSelection, pd.DataFrame]`

- [ ] **Step 1：写公式失败测试**

```python
import numpy as np
import pytest

from roughness.scheme1_physics.physics import exact_ra_um, word_ra_um


def test_word_formula_converts_mm_to_um():
    assert word_ra_um(0.04, 0.2) == pytest.approx(
        1000.0 * 0.04**2 / (32.0 * 0.2)
    )


def test_exact_formula_rejects_invalid_domain():
    with pytest.raises(ValueError, match="fz_mm must satisfy"):
        exact_ra_um(0.21, 0.10)


def test_exact_converges_to_word_for_small_ratio():
    exact = exact_ra_um(1e-4, 0.8)
    approx = word_ra_um(1e-4, 0.8)
    assert exact == pytest.approx(approx, rel=1e-7)


def test_formula_arrays_are_finite():
    fz = np.array([0.02, 0.04, 0.08])
    assert np.isfinite(exact_ra_um(fz, 0.2)).all()
    assert np.isfinite(word_ra_um(fz, 0.2)).all()
```

- [ ] **Step 2：运行公式测试并确认失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1_physics\test_physics.py -v
```

Expected: FAIL，提示目标函数不存在。

- [ ] **Step 3：实现公式**

```python
import numpy as np


def word_ra_um(fz_mm, radius_mm):
    fz = np.asarray(fz_mm, dtype=np.float64)
    radius = float(radius_mm)
    if radius <= 0 or np.any(fz < 0):
        raise ValueError("radius_mm must be positive and fz_mm non-negative")
    result = 1000.0 * fz**2 / (32.0 * radius)
    return float(result) if result.ndim == 0 else result


def exact_ra_um(fz_mm, radius_mm):
    fz = np.asarray(fz_mm, dtype=np.float64)
    radius = float(radius_mm)
    if radius <= 0:
        raise ValueError("radius_mm must be positive")
    if np.any((fz < 0) | (fz > 2.0 * radius)):
        raise ValueError("fz_mm must satisfy 0 <= fz_mm <= 2 * radius_mm")
    radicand = np.maximum(radius**2 - (fz / 2.0) ** 2, 0.0)
    result = 1000.0 * (radius - np.sqrt(radicand)) / 4.0
    return float(result) if result.ndim == 0 else result
```

`physical_ra_um`只接受 `formula in {"word","exact"}`，其他值抛出 `ValueError`。

- [ ] **Step 4：写尺度选择失败测试**

```python
def test_scale_selection_never_receives_outer_test_rows(
    miniature_manifest, fixed_folds
):
    train, test, selected, audit = build_physics_outer_frames(
        miniature_manifest,
        fixed_folds,
        outer_fold=0,
        seed=20260723,
        formula="word",
        candidates_mm=(0.1, 0.2),
        inner_splits=2,
    )
    train_ids = set(train["sample_id"])
    test_ids = set(test["sample_id"])
    assert train_ids.isdisjoint(test_ids)
    assert set(audit["source_sample_ids"].str.split("|").explode()) <= train_ids
    assert selected.radius_mm in {0.1, 0.2}


def test_same_selection_can_be_shared_by_three_models(
    miniature_manifest, fixed_folds
):
    train = miniature_manifest.merge(
        fixed_folds, on="sample_id", validate="one_to_one"
    )
    train = train[train["fold"] != 0].reset_index(drop=True)
    selected, audit = select_effective_scale(
        train,
        formula="exact",
        candidates_mm=(0.075, 0.1, 0.2),
        inner_splits=2,
        outer_fold=0,
        seed=20260723,
    )
    assert selected.formula == "exact"
    assert set(audit["candidate_radius_mm"]) == {0.075, 0.1, 0.2}
```

- [ ] **Step 5：实现选择记录和外层帧构造**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleSelection:
    formula: str
    outer_fold: int
    seed: int
    radius_mm: float
    inner_weighted_mae: float
    source_sample_ids: tuple[str, ...]
```

`select_effective_scale`必须：

1. 调用 `make_group_inner_splits`；
2. 对每个候选在每个内层验证子集计算原始公式预测；
3. 汇总所有内层验证行的加权绝对误差；
4. 按 `(weighted_mae, radius_mm)` 排序，误差相同时选择较小半径；
5. 返回每个候选的 `candidate_radius_mm,weighted_mae,source_sample_ids` 审计表。

审计表的 `source_sample_ids` 使用按字典序排序后以 `|` 连接的字符串，确保测试和fingerprint稳定。

`build_physics_outer_frames`必须：

- 按固定外层fold切分；
- 仅将外层训练帧传给 `select_effective_scale`；
- 在训练帧和测试帧新增原始微米列 `base_ra`；
- 返回的选择对象写明 `outer_fold` 与 `seed`。

- [ ] **Step 6：运行公式和尺度选择测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1_physics\test_physics.py -v
```

Expected: PASS。

- [ ] **Step 7：任务检查点**

若Git以后可用，建议提交：

```powershell
git add src/roughness/scheme1_physics/physics.py tests/scheme1_physics/test_physics.py
git commit -m "feat: add paired roughness physics formulas"
```

---

### Task 3：准备逐折标准化数据和DataLoader

**Files:**
- Create: `src/roughness/scheme1_physics/data.py`
- Create: `tests/scheme1_physics/test_data.py`

**Interfaces:**
- Consumes:
  - `ScaleSelection`
  - `SegmentBagDataset`
  - `collate_segment_bags`
  - 源方案1的 `window_index.csv` 与逐折 `channel_stats.json`
- Produces:
  - `InputScaler`
  - `PreparedPhysicsFold`
  - `prepare_physics_fold(config, formula, outer_fold, seed) -> PreparedPhysicsFold`
  - `build_physics_loaders(prepared, config, *, batch_size, validation_fraction, seed, device) -> LoaderBundle`

- [ ] **Step 1：写训练折标准化失败测试**

```python
import numpy as np

from roughness.scheme1_physics.data import fit_input_scaler


def test_scaler_uses_only_outer_train_rows(train_frame, outer_test_frame):
    scaler = fit_input_scaler(train_frame)
    transformed = scaler.transform(outer_test_frame)
    assert scaler.source_sample_ids == tuple(train_frame["sample_id"].astype(str))
    assert "base_ra_scaled" in transformed
    assert np.isfinite(
        transformed[
            ["n_rpm", "fz_mm_per_tooth", "ap_mm", "base_ra_scaled"]
        ].to_numpy()
    ).all()


def test_raw_base_ra_is_not_overwritten(train_frame):
    scaler = fit_input_scaler(train_frame)
    transformed = scaler.transform(train_frame)
    assert np.array_equal(transformed["base_ra"], train_frame["base_ra"])
```

- [ ] **Step 2：运行数据测试并确认失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1_physics\test_data.py -v
```

Expected: FAIL，提示 `fit_input_scaler` 不存在。

- [ ] **Step 3：实现标准化接口**

```python
from dataclasses import dataclass


PROCESS_COLUMNS = ("n_rpm", "fz_mm_per_tooth", "ap_mm")


@dataclass(frozen=True)
class InputScaler:
    mean: dict[str, float]
    scale: dict[str, float]
    source_sample_ids: tuple[str, ...]

    def transform(self, frame):
        result = frame.copy()
        for column in PROCESS_COLUMNS:
            result[column] = (
                result[column] - self.mean[column]
            ) / self.scale[column]
        result["base_ra_scaled"] = (
            result["base_ra"] - self.mean["base_ra"]
        ) / self.scale["base_ra"]
        return result
```

`fit_input_scaler`对 `PROCESS_COLUMNS + ("base_ra",)` 使用训练帧均值和总体标准差 `ddof=0`；零标准差替换为1.0；元数据记录完整训练 `sample_id`。

- [ ] **Step 3a：在conftest加入Task 3夹具**

```python
@pytest.fixture
def physics_config(tiny_workspace):
    from roughness.scheme1_physics.config import (
        load_scheme1_physics_config,
    )
    return load_scheme1_physics_config(tiny_workspace.physics_config_path)


@pytest.fixture
def train_frame(miniature_manifest, fixed_folds):
    frame = miniature_manifest.merge(fixed_folds, on="sample_id")
    frame = frame[frame["fold"] != 0].reset_index(drop=True)
    frame["base_ra"] = 1000.0 * frame["fz_mm_per_tooth"] ** 2 / (32 * 0.2)
    return frame


@pytest.fixture
def outer_test_frame(miniature_manifest, fixed_folds):
    frame = miniature_manifest.merge(fixed_folds, on="sample_id")
    frame = frame[frame["fold"] == 0].reset_index(drop=True)
    frame["base_ra"] = 1000.0 * frame["fz_mm_per_tooth"] ** 2 / (32 * 0.2)
    return frame
```

- [ ] **Step 4：实现折数据对象**

```python
@dataclass(frozen=True)
class PreparedPhysicsFold:
    formula: str
    outer_fold: int
    seed: int
    selection: ScaleSelection
    selection_audit: pd.DataFrame
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    train_scaled: pd.DataFrame
    test_scaled: pd.DataFrame
    scaler: InputScaler
```

`prepare_physics_fold`调用Task 2的 `build_physics_outer_frames`，随后只在 `train_raw` 上拟合 `InputScaler`。它不得读取M0预测。

在conftest加入：

```python
@pytest.fixture
def prepared_fold(physics_config):
    from roughness.scheme1_physics.data import prepare_physics_fold
    return prepare_physics_fold(
        physics_config, formula="word", outer_fold=0, seed=20260723
    )
```

- [ ] **Step 5：构造Dataset和分组Loader**

`build_physics_loaders`复用：

```python
SegmentBagDataset(
    frame,
    window_index,
    channel_stats,
    horizontal_mode="raw",
    physics_columns=("base_ra_scaled",),
    base_ra_column="base_ra",
    cache_signals=True,
)
```

训练/验证索引必须调用：

```python
make_group_train_validation_split(
    prepared.train_scaled,
    validation_fraction=validation_fraction,
    seed=seed,
)
```

并断言训练、验证、外层测试的 `group_id` 两两不相交。

`LoaderBundle`定义为：

```python
@dataclass(frozen=True)
class LoaderBundle:
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    train_groups: frozenset[str]
    validation_groups: frozenset[str]
    test_groups: frozenset[str]
```

- [ ] **Step 6：写Loader分组测试并运行**

```python
def test_loader_groups_are_disjoint(prepared_fold, physics_config):
    loaders = build_physics_loaders(
        prepared_fold,
        physics_config,
        batch_size=2,
        validation_fraction=0.25,
        seed=20260723,
        device="cpu",
    )
    assert loaders.train_groups.isdisjoint(loaders.validation_groups)
    assert loaders.train_groups.isdisjoint(loaders.test_groups)
    assert loaders.validation_groups.isdisjoint(loaders.test_groups)
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1_physics\test_data.py -v
```

Expected: PASS。

- [ ] **Step 7：任务检查点**

若Git以后可用，建议提交：

```powershell
git add src/roughness/scheme1_physics/data.py tests/scheme1_physics/test_data.py
git commit -m "feat: prepare leakage-safe physics fold data"
```

---

### Task 4：实现普通残差和物理修正门控模型

**Files:**
- Create: `src/roughness/scheme1_physics/models.py`
- Create: `tests/scheme1_physics/test_models.py`

**Interfaces:**
- Consumes:
  - `roughness.scheme1.models.CNNEncoder`
  - `roughness.scheme1.models.ModelOutput`
  - `roughness.scheme1.models.masked_mean_pool`
- Produces:
  - `PhysicsResidualRegressor(mode: Literal["ordinary","gated"], dropout: float)`
  - `load_ordinary_into_gated(gated, ordinary_state) -> None`
  - `freeze_residual_for_gate(model) -> dict[str, torch.Tensor]`
  - `assert_frozen_state_unchanged(model, snapshot) -> None`

- [ ] **Step 1：写模型恒等式失败测试**

```python
import torch

from roughness.scheme1_physics.models import PhysicsResidualRegressor


def make_batch():
    return {
        "signal": torch.randn(2, 3, 3, 256),
        "window_mask": torch.ones(2, 3, dtype=torch.bool),
        "process": torch.randn(2, 3),
        "physics": torch.randn(2, 1),
        "base_ra": torch.tensor([0.2, 0.4]),
    }


def test_ordinary_prediction_is_physics_plus_residual():
    batch = make_batch()
    model = PhysicsResidualRegressor(mode="ordinary", dropout=0.0)
    output = model(**batch)
    assert torch.allclose(
        output.prediction, batch["base_ra"] + output.residual
    )


def test_gated_prediction_uses_one_minus_gate():
    batch = make_batch()
    model = PhysicsResidualRegressor(mode="gated", dropout=0.0)
    output = model(**batch)
    assert torch.all((0 <= output.gate) & (output.gate <= 1))
    assert torch.allclose(
        output.prediction,
        batch["base_ra"] + (1.0 - output.gate) * output.residual,
    )
```

测试实现时只调用一次 `make_batch()` 并复用同一字典，避免随机输入不同导致假失败。

- [ ] **Step 2：运行模型测试并确认失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1_physics\test_models.py -v
```

Expected: FAIL，提示 `PhysicsResidualRegressor` 不存在。

- [ ] **Step 3：实现模型**

```python
from typing import Literal

import torch
from torch import nn

from roughness.scheme1.models import CNNEncoder, ModelOutput, masked_mean_pool


class PhysicsResidualRegressor(nn.Module):
    def __init__(
        self,
        mode: Literal["ordinary", "gated"],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if mode not in {"ordinary", "gated"}:
            raise ValueError("mode must be ordinary or gated")
        self.mode = mode
        self.encoder = CNNEncoder()
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim + 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.gate_layer = nn.Linear(4, 1) if mode == "gated" else None
        self._gate_only_training = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self._gate_only_training:
            self.encoder.eval()
            self.residual_mlp.eval()
            self.gate_layer.train(mode)
        return self

    def encode_segment(self, signal, window_mask):
        batch, windows, channels, samples = signal.shape
        embedded = self.encoder(
            signal.reshape(batch * windows, channels, samples)
        ).reshape(batch, windows, -1)
        return masked_mean_pool(embedded, window_mask)

    def forward(
        self,
        signal,
        window_mask,
        process,
        physics,
        base_ra,
    ) -> ModelOutput:
        if process.shape[-1] != 3 or physics.shape[-1] != 1:
            raise ValueError("expected process_dim=3 and physics_dim=1")
        context = torch.cat([process, physics], dim=-1)
        embedding = self.encode_segment(signal, window_mask)
        residual = self.residual_mlp(
            torch.cat([embedding, context], dim=-1)
        ).squeeze(-1)
        if self.mode == "ordinary":
            return ModelOutput(
                prediction=base_ra + residual,
                residual=residual,
            )
        gate = torch.sigmoid(self.gate_layer(context).squeeze(-1))
        return ModelOutput(
            prediction=base_ra + (1.0 - gate) * residual,
            residual=residual,
            gate=gate,
        )
```

- [ ] **Step 4：写冻结与权重转移失败测试**

```python
def test_gate_training_freezes_encoder_and_residual_head():
    ordinary = PhysicsResidualRegressor("ordinary", dropout=0.0)
    gated = PhysicsResidualRegressor("gated", dropout=0.0)
    load_ordinary_into_gated(gated, ordinary.state_dict())
    snapshot = freeze_residual_for_gate(gated)
    assert gated.gate_layer.weight.requires_grad
    assert all(
        not parameter.requires_grad
        for name, parameter in gated.named_parameters()
        if not name.startswith("gate_layer.")
    )
    assert_frozen_state_unchanged(gated, snapshot)
```

- [ ] **Step 5：实现权重转移和冻结校验**

`load_ordinary_into_gated`必须仅允许缺失 `gate_layer.weight` 与 `gate_layer.bias`：

```python
missing, unexpected = gated.load_state_dict(ordinary_state, strict=False)
if set(missing) != {"gate_layer.weight", "gate_layer.bias"} or unexpected:
    raise ValueError("ordinary checkpoint is incompatible with gated model")
```

`freeze_residual_for_gate`将非门控参数 `requires_grad_(False)`，门控参数 `requires_grad_(True)`，设置 `model._gate_only_training = True`，并返回非门控参数的CPU克隆。覆盖后的 `train()` 必须让冻结的CNN和残差MLP始终保持evaluation模式，从而关闭Dropout；只让门控层进入训练模式。`assert_frozen_state_unchanged`逐项使用 `torch.equal`，任何变化均抛出 `AssertionError`。

增加测试：

```python
def test_gate_only_mode_disables_frozen_dropout():
    model = PhysicsResidualRegressor("gated", dropout=0.3)
    freeze_residual_for_gate(model)
    model.train()
    assert not model.encoder.training
    assert not model.residual_mlp.training
    assert model.gate_layer.training
```

- [ ] **Step 6：运行模型测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1_physics\test_models.py -v
```

Expected: PASS。

- [ ] **Step 7：任务检查点**

若Git以后可用，建议提交：

```powershell
git add src/roughness/scheme1_physics/models.py tests/scheme1_physics/test_models.py
git commit -m "feat: add interpretable physics residual gate"
```

---

### Task 5：生成物理OOF并训练普通残差模型

**Files:**
- Create: `src/roughness/scheme1_physics/training.py`
- Create: `tests/scheme1_physics/test_training.py`

**Interfaces:**
- Consumes:
  - `prepare_physics_fold`
  - `build_physics_loaders`
  - `PhysicsResidualRegressor`
  - `roughness.scheme1.training.train_model`
  - `roughness.scheme1.training.compute_run_fingerprint`
- Produces:
  - `physics_model_formula(model_name: str) -> str`
  - `physics_model_mode(model_name: str) -> str`
  - `write_physics_baseline_oof(config, force=False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
  - `physics_run_fingerprint(...) -> str`
  - `train_physics_fold(model_name, outer_fold, seed, config, ...) -> PhysicsFoldRunResult`

在 `tests/scheme1_physics/conftest.py` 增加：

```python
@pytest.fixture
def tiny_physics_config(physics_config):
    return physics_config
```

- [ ] **Step 1：写模型映射和物理OOF失败测试**

```python
import pytest

from roughness.scheme1_physics.training import (
    physics_model_formula,
    physics_model_mode,
)


@pytest.mark.parametrize(
    ("model", "formula", "mode"),
    [
        ("PW0", "word", "baseline"),
        ("PW1", "word", "ordinary"),
        ("PW2", "word", "gated"),
        ("PE0", "exact", "baseline"),
        ("PE1", "exact", "ordinary"),
        ("PE2", "exact", "gated"),
    ],
)
def test_model_mapping_is_fixed(model, formula, mode):
    assert physics_model_formula(model) == formula
    assert physics_model_mode(model) == mode


def test_baseline_oof_has_all_rows_once_per_model_seed(
    tiny_physics_config,
):
    predictions, selections, audits = write_physics_baseline_oof(
        tiny_physics_config, force=True
    )
    assert set(predictions["model"]) == {"PW0", "PE0"}
    assert not predictions.duplicated(["sample_id", "model", "seed"]).any()
    assert set(selections["formula"]) == {"word", "exact"}
    assert set(audits["candidate_radius_mm"]) == set(
        tiny_physics_config.source.re_candidates_mm
    )
```

- [ ] **Step 2：运行训练测试并确认失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1_physics\test_training.py -v
```

Expected: FAIL，提示训练接口不存在。

- [ ] **Step 3：实现物理OOF**

`write_physics_baseline_oof`遍历：

```python
for formula, model_name in (("word", "PW0"), ("exact", "PE0")):
    for outer_fold in range(5):
        for seed in config.source.seeds:
            prepared = prepare_physics_fold(
                config, formula, outer_fold, seed
            )
```

每条OOF记录至少包含：

```text
sample_id,group_id,model,formula,radius_mm,fold,seed,
y_true,y_pred,base_ra,sample_weight
```

并分别写入：

```text
outputs/scheme1_physics/physics/oof_predictions.csv
outputs/scheme1_physics/physics/selections.csv
outputs/scheme1_physics/physics/candidate_audit.csv
```

- [ ] **Step 4：实现普通残差前向和运行指纹**

```python
def physics_forward(model, batch):
    return model(
        signal=batch["signal"],
        window_mask=batch["window_mask"],
        process=batch["process"],
        physics=batch["physics"],
        base_ra=batch["base_ra"],
    ).prediction
```

`physics_run_fingerprint`必须哈希：

- manifest、folds、window index；
- 当前fold的channel stats；
- 当前公式/折/种子的选择记录；
- 模型名、训练轮数、batch size、验证比例；
- 学习率、权重衰减、dropout；
- 对门控模型额外加入对应普通残差checkpoint。

- [ ] **Step 5：实现普通残差单折训练**

`train_physics_fold`先只接受 `PW1/PE1`。复用方案1的 `train_model`，但保存到：

```text
outputs/scheme1_physics/neural/{model}/fold_{fold}/seed_{seed}/
```

OOF每行增加：

```text
residual,base_ra,formula,radius_mm
```

`PhysicsFoldRunResult`字段固定为：

```python
@dataclass(frozen=True)
class PhysicsFoldRunResult:
    model_name: str
    formula: str
    outer_fold: int
    seed: int
    radius_mm: float
    best_epoch: int
    best_validation_mae: float
    checkpoint_path: Path
    log_path: Path
    oof_path: Path
```

方案1的通用 `train_model` 写完checkpoint后，`train_physics_fold`必须重新读取并补充 `model,formula,outer_fold,seed,radius_mm,run_fingerprint`，再原子替换同一路径，使Task 6能够严格验证父checkpoint来源。

- [ ] **Step 6：写普通残差微型训练测试**

```python
def test_train_ordinary_fold_writes_complete_oof(
    tiny_physics_config,
):
    result = train_physics_fold(
        "PW1",
        outer_fold=0,
        seed=20260723,
        config=tiny_physics_config,
        device=torch.device("cpu"),
        max_epochs=1,
        batch_size=2,
    )
    oof = pd.read_csv(result.oof_path)
    assert set(
        ["base_ra", "residual", "formula", "radius_mm"]
    ) <= set(oof.columns)
    assert oof["formula"].eq("word").all()
    assert len(oof) > 0
```

- [ ] **Step 7：运行普通残差训练测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1_physics\test_training.py -v
```

Expected: PASS。

- [ ] **Step 8：任务检查点**

若Git以后可用，建议提交：

```powershell
git add src/roughness/scheme1_physics/training.py tests/scheme1_physics/test_training.py
git commit -m "feat: train physical baseline residual models"
```

---

### Task 6：实现两阶段门控训练与可恢复运行

**Files:**
- Modify: `src/roughness/scheme1_physics/training.py`
- Modify: `tests/scheme1_physics/test_training.py`

**Interfaces:**
- Consumes:
  - PW2加载同折同种子的PW1 checkpoint
  - PE2加载同折同种子的PE1 checkpoint
- Produces:
  - `ordinary_parent_model(model_name: str) -> str`
  - `train_gated_physics_fold(...) -> PhysicsFoldRunResult`
  - `completed_physics_run_matches(run_dir, expected_fingerprint) -> bool`
  - OOF列 `gate` 与冻结参数审计文件

在 `tests/scheme1_physics/test_training.py` 中加入真实的1轮父模型夹具：

```python
@pytest.fixture
def pe1_checkpoint(tiny_physics_config):
    result = train_physics_fold(
        "PE1",
        outer_fold=0,
        seed=20260723,
        config=tiny_physics_config,
        device=torch.device("cpu"),
        max_epochs=1,
        batch_size=2,
    )
    return result.checkpoint_path


@pytest.fixture
def trained_pw2_result(tiny_physics_config):
    parent = train_physics_fold(
        "PW1",
        outer_fold=0,
        seed=20260723,
        config=tiny_physics_config,
        device=torch.device("cpu"),
        max_epochs=1,
        batch_size=2,
    )
    return train_gated_physics_fold(
        "PW2",
        outer_fold=0,
        seed=20260723,
        config=tiny_physics_config,
        parent_checkpoint=parent.checkpoint_path,
        device=torch.device("cpu"),
        max_epochs=1,
        batch_size=2,
    )
```

- [ ] **Step 1：写父模型和来源错误测试**

```python
def test_gate_parent_mapping():
    assert ordinary_parent_model("PW2") == "PW1"
    assert ordinary_parent_model("PE2") == "PE1"
    with pytest.raises(ValueError, match="gated model"):
        ordinary_parent_model("PW1")


def test_gate_rejects_checkpoint_from_other_formula(
    tiny_physics_config, pe1_checkpoint
):
    with pytest.raises(ValueError, match="formula mismatch"):
        train_gated_physics_fold(
            "PW2",
            outer_fold=0,
            seed=20260723,
            config=tiny_physics_config,
            parent_checkpoint=pe1_checkpoint,
            max_epochs=1,
            device=torch.device("cpu"),
        )
```

- [ ] **Step 2：实现门控checkpoint校验**

普通残差checkpoint必须扩展元数据：

```text
model,formula,outer_fold,seed,radius_mm,run_fingerprint,model_state
```

门控加载前逐项校验：

```python
expected = {
    "model": ordinary_parent_model(model_name),
    "formula": physics_model_formula(model_name),
    "outer_fold": int(outer_fold),
    "seed": int(seed),
    "radius_mm": float(prepared.selection.radius_mm),
}
```

任何不一致都拒绝训练，不允许按键名“尽量加载”。

- [ ] **Step 3：实现仅训练门控层**

```python
ordinary_payload = torch.load(parent_checkpoint, map_location="cpu")
model = PhysicsResidualRegressor(
    mode="gated",
    dropout=config.residual_dropout,
)
load_ordinary_into_gated(model, ordinary_payload["model_state"])
frozen_snapshot = freeze_residual_for_gate(model)
trainable = [
    parameter for parameter in model.parameters() if parameter.requires_grad
]
if {id(parameter) for parameter in trainable} != {
    id(model.gate_layer.weight),
    id(model.gate_layer.bias),
}:
    raise AssertionError("only gate parameters may be trainable")
```

训练完成后必须调用 `assert_frozen_state_unchanged`，并写入：

```text
frozen_parameter_audit.json
```

内容至少包括冻结参数数量、门控参数数量、父checkpoint指纹和 `unchanged: true`。

- [ ] **Step 4：写门控OOF测试**

```python
def test_gated_oof_contains_gate_and_identity(
    trained_pw2_result,
):
    oof = pd.read_csv(trained_pw2_result.oof_path)
    assert oof["gate"].between(0, 1).all()
    reconstructed = (
        oof["base_ra"] + (1.0 - oof["gate"]) * oof["residual"]
    )
    assert np.allclose(oof["y_pred"], reconstructed, atol=1e-6)
```

- [ ] **Step 5：实现断点续跑匹配**

`completed_physics_run_matches`只在以下条件全部成立时返回 `True`：

- `run_metadata.json`与 `oof_predictions.csv`存在；
- `status == "complete"`；
- `run_fingerprint`严格匹配；
- 对门控模型，`parent_checkpoint_fingerprint`严格匹配；
- OOF中模型、fold、seed、formula和radius与请求一致。

- [ ] **Step 6：运行两阶段训练测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1_physics\test_training.py -v
```

Expected: PASS，包括冻结参数未变化、跨公式checkpoint拒绝、门控恒等式和断点续跑测试。

- [ ] **Step 7：任务检查点**

若Git以后可用，建议提交：

```powershell
git add src/roughness/scheme1_physics/training.py tests/scheme1_physics/test_training.py
git commit -m "feat: add staged physics confidence gate training"
```

---

### Task 7：实现完整OOF校验、指标和预注册验收

**Files:**
- Create: `src/roughness/scheme1_physics/evaluation.py`
- Create: `tests/scheme1_physics/test_evaluation.py`

**Interfaces:**
- Consumes:
  - 六模型OOF
  - 源方案1 M0 OOF
  - `roughness.scheme1.evaluation.paired_group_bootstrap`
- Produces:
  - `validate_complete_physics_oof(frame, expected_sample_ids, models, seeds) -> None`
  - `physics_metric_rows(predictions) -> pd.DataFrame`
  - `assess_physics_comparisons(metrics, predictions, m0_predictions, config) -> dict`
  - `apply_secondary_metric_downgrade(result, reference_metrics, candidate_metrics) -> dict`
  - `select_final_candidate(acceptance, metrics) -> dict`

在 `tests/scheme1_physics/test_evaluation.py` 中显式提供合成夹具：

```python
@pytest.fixture
def complete_oof():
    rows = []
    models = ("PW0", "PW1", "PW2", "PE0", "PE1", "PE2")
    for index, sample_id in enumerate(("s0", "s1")):
        for model in models:
            rows.append(
                {
                    "sample_id": sample_id,
                    "group_id": f"g{index}",
                    "model": model,
                    "formula": "word" if model.startswith("PW") else "exact",
                    "radius_mm": 0.2,
                    "fold": index,
                    "seed": 20260723,
                    "y_true": 0.2 + 0.1 * index,
                    "y_pred": 0.21 + 0.1 * index,
                    "base_ra": 0.05,
                    "sample_weight": 1.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def comparison_metrics():
    rows = []
    for seed in (20260723, 20260724, 20260725):
        for fold in range(5):
            rows.extend(
                [
                    {
                        "model": "PW1",
                        "seed": seed,
                        "fold": fold,
                        "weighted_mae": 1.0 + 0.01 * fold,
                    },
                    {
                        "model": "PW2",
                        "seed": seed,
                        "fold": fold,
                        "weighted_mae": 0.98 + 0.005 * fold,
                    },
                ]
            )
    return pd.DataFrame(rows)
```

- [ ] **Step 1：写完整OOF失败测试**

```python
import pandas as pd
import pytest

from roughness.scheme1_physics.evaluation import (
    validate_complete_physics_oof,
)


def test_complete_oof_rejects_duplicate_sample_model_seed(complete_oof):
    duplicated = pd.concat([complete_oof, complete_oof.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        validate_complete_physics_oof(
            duplicated,
            expected_sample_ids=set(complete_oof["sample_id"]),
            models=["PW0", "PW1", "PW2", "PE0", "PE1", "PE2"],
            seeds=[20260723],
        )


def test_complete_oof_rejects_missing_model_rows(complete_oof):
    incomplete = complete_oof[complete_oof["model"] != "PE2"]
    with pytest.raises(ValueError, match="Incomplete OOF"):
        validate_complete_physics_oof(
            incomplete,
            expected_sample_ids=set(complete_oof["sample_id"]),
            models=["PW0", "PW1", "PW2", "PE0", "PE1", "PE2"],
            seeds=[20260723],
        )
```

- [ ] **Step 2：实现完整OOF校验和指标**

完整性主键固定为：

```text
sample_id,model,seed
```

每个模型/种子必须覆盖manifest中的全部586个 `sample_id`，并要求：

- `y_true,y_pred,sample_weight,base_ra,radius_mm`有限；
- `sample_weight > 0`；
- `PW*`公式列为 `word`，`PE*`公式列为 `exact`；
- 同一公式/折/种子的0/1/2模型半径完全相同。

`physics_metric_rows`按 `model,seed,fold`输出普通和加权的 `mae,rmse,r2`。

- [ ] **Step 3：写验收门槛失败测试**

```python
def test_gate_requires_one_percent_three_folds_and_stable_variance(
    comparison_metrics,
):
    result = assess_pair(
        comparison_metrics,
        candidate="PW2",
        reference="PW1",
        minimum_relative_improvement=0.01,
        minimum_fold_wins=3,
        require_all_seed_means=True,
        maximum_fold_std_relative_increase=0.20,
    )
    assert result["passed"]


def test_secondary_metric_conflict_downgrades_stable_to_exploratory():
    result = {"classification": "stable_effective"}
    downgraded = apply_secondary_metric_downgrade(
        result,
        reference_metrics={"weighted_rmse": 0.10, "weighted_r2": 0.80},
        candidate_metrics={"weighted_rmse": 0.104, "weighted_r2": 0.81},
    )
    assert downgraded["classification"] == "exploratory_increment"
    assert "rmse_conflict" in downgraded["downgrade_reasons"]
```

- [ ] **Step 4：实现比较矩阵**

必须生成以下机器可读比较：

```python
COMPARISONS = {
    "PW1_vs_PW0": ("PW1", "PW0", 0.05, 3, True, None),
    "PE1_vs_PE0": ("PE1", "PE0", 0.05, 3, True, None),
    "PW2_vs_PW1": ("PW2", "PW1", 0.01, 3, True, 0.20),
    "PE2_vs_PE1": ("PE2", "PE1", 0.01, 3, True, 0.20),
    "PE0_vs_PW0": ("PE0", "PW0", 0.01, 3, True, None),
    "PE1_vs_PW1": ("PE1", "PW1", 0.01, 3, True, None),
    "PE2_vs_PW2": ("PE2", "PW2", 0.01, 3, True, None),
}
```

元组中的布尔值表示“三个种子的折均MAE均与总体改善方向一致”，不是只检查三个种子合并后的总均值。

公式优劣允许任一方向，因此同时计算PE优于PW与PW优于PE；只有一个方向满足1%、3/5折、三个种子平均同方向和Bootstrap区间排除零时才下结论。

- [ ] **Step 5：实现M0最终分级**

对 `PW1,PW2,PE1,PE2`逐一与M0比较：

- `<3%`或折/种子方向反复：`no_stable_increment`；
- `>=3%`、至少3/5折、三个种子平均改善：`exploratory_increment`；
- `>=5%`、至少4/5折、三个种子分别改善、Bootstrap `ci_low > 0`：`stable_effective`。

候选模型的Bootstrap使用三个种子的区域级平均预测；M0使用源方案1固定OOF预测。抽样单位必须为 `group_id`，重复10,000次。

如果候选MAE通过，但RMSE相对M0恶化超过3%或加权 \(R^2\) 低于M0，结论降级一级。

- [ ] **Step 6：实现最终候选选择**

`select_final_candidate`只在通过普通残差或门控阶梯的模型中选择：

1. 门控未过对应普通残差门槛时，排除PW2或PE2；
2. 剩余模型按 `classification` 的 `stable > exploratory > none` 排序；
3. 同等级按三个种子折均值加权MAE升序；
4. 完全相同时按固定顺序 `PW1,PE1,PW2,PE2`，避免运行顺序影响结果。

- [ ] **Step 7：运行评估测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1_physics\test_evaluation.py -v
```

Expected: PASS。

- [ ] **Step 8：任务检查点**

若Git以后可用，建议提交：

```powershell
git add src/roughness/scheme1_physics/evaluation.py tests/scheme1_physics/test_evaluation.py
git commit -m "feat: add preregistered physics model acceptance"
```

---

### Task 8：实现报告、图表和方法限制

**Files:**
- Create: `src/roughness/scheme1_physics/reporting.py`
- Create: `tests/scheme1_physics/test_reporting.py`

**Interfaces:**
- Consumes: 六模型OOF、M0 OOF、manifest、selection、acceptance。
- Produces:
  - `write_formula_difference_report(...) -> dict[str, Path]`
  - `write_gate_reports(...) -> dict[str, Path]`
  - `write_physics_prediction_figures(...) -> dict[str, Path]`
  - `write_physics_error_breakdowns(...) -> dict[str, Path]`
  - `write_method_notes(output_dir) -> Path`

在 `tests/scheme1_physics/test_reporting.py` 中显式提供：

```python
@pytest.fixture
def formula_predictions():
    return pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "radius_mm": [0.2, 0.2],
            "fz_mm_per_tooth": [0.02, 0.04],
            "exact_um": [0.06251, 0.25031],
            "word_um": [0.06250, 0.25000],
        }
    )


@pytest.fixture
def gated_oof():
    return pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s0", "s1"],
            "model": ["PW2", "PW2", "PE2", "PE2"],
            "gate": [0.2, 0.8, 0.3, 0.7],
            "n_rpm": [6000, 7000, 6000, 7000],
            "fz_mm_per_tooth": [0.02, 0.04, 0.02, 0.04],
            "ap_mm": [0.5, 1.0, 0.5, 1.0],
            "base_ra": [0.05, 0.1, 0.051, 0.101],
        }
    )
```

- [ ] **Step 1：写报告失败测试**

```python
def test_formula_report_contains_absolute_and_relative_error(
    formula_predictions, tmp_path
):
    paths = write_formula_difference_report(
        formula_predictions, tmp_path
    )
    table = pd.read_csv(paths["table"])
    assert {
        "sample_id",
        "radius_mm",
        "exact_um",
        "word_um",
        "absolute_difference_um",
        "relative_difference",
    } <= set(table.columns)


def test_gate_report_only_uses_gated_models(gated_oof, tmp_path):
    paths = write_gate_reports(gated_oof, tmp_path)
    summary = pd.read_csv(paths["summary"])
    assert set(summary["model"]) == {"PW2", "PE2"}
    assert summary["gate_min"].between(0, 1).all()
    assert summary["gate_max"].between(0, 1).all()
```

- [ ] **Step 2：实现公式差异报告**

输出分为两个互不混淆的层次：

```text
evaluation/formula_difference_by_candidate.csv
evaluation/formula_difference_by_candidate_summary.csv
evaluation/formula_selected_prediction_difference.csv
evaluation/figures/formula_relative_difference.png
```

`formula_difference_by_candidate.csv`对每个样本和每个共同候选半径直接计算精确式与Word式，用于隔离纯公式近似误差；列为 `sample_id,radius_mm,fz_mm_per_tooth,exact_um,word_um,absolute_difference_um,relative_difference`。

`formula_selected_prediction_difference.csv`比较PE0与PW0各自折内选尺度后的预测，必须分别保留 `exact_radius_mm` 与 `word_radius_mm`，用于评价完整选模流程，不得解释为纯公式误差。

相对差异分母使用 `max(abs(exact_um), 1e-12)`，共同候选表按半径与进给量汇总；选尺度表按外层折和种子汇总。

- [ ] **Step 3：实现门控报告**

输出：

```text
evaluation/gates/gate_summary.csv
evaluation/gates/gate_by_process_bin.csv
evaluation/figures/gate_distribution_PW2.png
evaluation/figures/gate_distribution_PE2.png
evaluation/figures/gate_vs_process.png
```

分箱边界只根据manifest中的固定工艺取值或全数据可见的离散类别生成；不得利用目标值确定分箱。

- [ ] **Step 4：复用并扩展预测与误差图**

调用方案1的 `write_prediction_figures` 与 `write_error_breakdowns`，另外输出：

- 六模型＋M0预测—实测图；
- 六模型折/种子MAE图；
- 最终候选相对M0的组级误差差值；
- 按 `n_rpm,fz_mm_per_tooth,ap_mm,version,ra区间` 的误差表。

粗糙度区间必须在报告函数中以固定边界或等宽边界预先定义，不按候选模型误差自适应选择。

- [ ] **Step 5：写方法限制JSON**

`method_notes.json`固定包含：

```json
{
  "effective_radius": "Candidate radii are sensitivity scales, not measured tool radii.",
  "formula_relation": "The Word formula is the small-feed approximation of the exact circular formula.",
  "gate_meaning": "Gate values are internal residual-shrinkage coefficients, not calibrated physical probabilities.",
  "causal_limit": "No exact matched no-physics architecture was added, so independent causal attribution to the formula is not claimed.",
  "sensor_direction": "Horizontal channels retain the source Scheme 1 raw ordering and are not relabeled as measured X/Y directions."
}
```

- [ ] **Step 6：运行报告测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1_physics\test_reporting.py -v
```

Expected: PASS，且所有PNG非空。

- [ ] **Step 7：任务检查点**

若Git以后可用，建议提交：

```powershell
git add src/roughness/scheme1_physics/reporting.py tests/scheme1_physics/test_reporting.py
git commit -m "feat: report physics formula and gate behavior"
```

---

### Task 9：实现CLI编排、断点续跑和最终产物

**Files:**
- Create: `src/roughness/scheme1_physics/cli.py`
- Create: `tests/scheme1_physics/test_cli.py`
- Modify: `src/roughness/scheme1_physics/evaluation.py`

**Interfaces:**
- Consumes: Tasks 1-8全部公共接口。
- Produces:
  - `roughness-scheme1-physics prepare`
  - `roughness-scheme1-physics baselines`
  - `roughness-scheme1-physics train`
  - `roughness-scheme1-physics evaluate`
  - `roughness-scheme1-physics run`
  - `ordered_neural_models(requested: list[str]) -> list[str]`

- [ ] **Step 1：写CLI解析失败测试**

```python
from roughness.scheme1_physics.cli import build_parser


def test_train_cli_defaults_to_four_neural_models():
    args = build_parser().parse_args(
        ["train", "--config", "configs/scheme1_physics.yaml"]
    )
    assert args.models == "PW1,PW2,PE1,PE2"
    assert args.folds == "0,1,2,3,4"
    assert args.seeds == "20260723,20260724,20260725"
    assert args.batch_size == 4
```

- [ ] **Step 2：实现CLI子命令**

`prepare`：

- 加载配置；
- 验证源方案1产物；
- 写 `run_manifest.json`；
- 验证manifest与fold；
- 不复制大型信号文件。

`baselines`：

- 运行PW0/PE0；
- 写selection、candidate audit和物理OOF；
- `--resume`时只复用fingerprint一致的完整产物。

`train`：

- 默认模型 `PW1,PW2,PE1,PE2`；
- 对每个公式先跑普通残差，再跑门控；
- 用户即使传入 `PW2`，若对应PW1 checkpoint缺失也应先明确报错，不自动用其他checkpoint；
- 支持 `--folds`、`--seeds`、`--max-epochs`、`--batch-size`、`--resume`。

`evaluate`：

- 合并物理与神经OOF；
- 校验完整性；
- 生成 `summary_metrics.csv`、`paired_bootstrap.csv`、`comparisons.json`、`acceptance.json`；
- 生成Task 8全部报告。

`run`严格按：

```text
prepare → baselines → PW1 → PE1 → PW2 → PE2 → evaluate
```

- [ ] **Step 3：写门控顺序测试**

```python
def test_run_orders_ordinary_before_gated():
    assert ordered_neural_models(
        ["PW2", "PE2", "PE1", "PW1"]
    ) == ["PW1", "PE1", "PW2", "PE2"]


def test_requested_gate_keeps_required_parent():
    assert ordered_neural_models(["PW2"]) == ["PW1", "PW2"]
    assert ordered_neural_models(["PE2"]) == ["PE1", "PE2"]
```

`ordered_neural_models`只接受 `PW1,PW2,PE1,PE2`，去重后按依赖拓扑排序；请求门控时自动包含其普通残差父模型。CLI `run`与 `train`均调用该函数。

- [ ] **Step 4：实现聚合OOF**

当且仅当fold为 `0,1,2,3,4` 时写正式文件：

```text
outputs/scheme1_physics/oof_predictions.csv
outputs/scheme1_physics/metrics.csv
```

预期正式OOF行数：

\[
586\text{ samples}\times6\text{ models}\times3\text{ seeds}
=10548.
\]

部分fold只写：

```text
oof_predictions_partial.csv
```

- [ ] **Step 5：实现机器可读验收文件**

`acceptance.json`顶层键固定为：

```json
{
  "formula_comparisons": {},
  "residual_gates": {},
  "gating_gates": {},
  "m0_comparisons": {},
  "bootstrap": {},
  "selected_model": {},
  "scheme1_physics_conclusion": ""
}
```

`scheme1_physics_conclusion`只允许：

```text
stable_effective
exploratory_increment
no_stable_increment
```

- [ ] **Step 6：运行CLI测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1_physics\test_cli.py -v
```

Expected: PASS。

- [ ] **Step 7：任务检查点**

若Git以后可用，建议提交：

```powershell
git add src/roughness/scheme1_physics/cli.py src/roughness/scheme1_physics/evaluation.py tests/scheme1_physics/test_cli.py
git commit -m "feat: orchestrate scheme1 physics experiments"
```

---

### Task 10：全套回归、微型冒烟和正式运行

**Files:**
- Modify only if verification exposes a defect:
  - `src/roughness/scheme1_physics/*.py`
  - `tests/scheme1_physics/*.py`
- Generate:
  - `outputs/scheme1_physics/**`

**Interfaces:**
- Consumes: 完整CLI。
- Produces: 经验证的正式PW/PE物理补充分支结果。

- [ ] **Step 1：运行新增测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1_physics -q
```

Expected: 全部PASS，无warning导致的失败。

- [ ] **Step 2：运行原方案1回归测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1 -q
```

Expected: 原方案1全部PASS；已有83项全套测试不得减少。

- [ ] **Step 3：运行项目全套测试**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: 全部PASS。

- [ ] **Step 4：确认原方案1产物未被修改**

在微型运行前后记录以下文件SHA-256：

```powershell
Get-FileHash outputs\scheme1\neural\oof_predictions.csv -Algorithm SHA256
Get-FileHash outputs\scheme1\evaluation\acceptance.json -Algorithm SHA256
Get-FileHash outputs\scheme1\classic\oof_predictions.csv -Algorithm SHA256
```

Expected: 前后哈希完全一致。

- [ ] **Step 5：运行单折单种子微型冒烟**

Run:

```powershell
.\.venv\Scripts\python.exe -m roughness.scheme1_physics.cli prepare --config configs\scheme1_physics.yaml
.\.venv\Scripts\python.exe -m roughness.scheme1_physics.cli baselines --config configs\scheme1_physics.yaml --folds 0 --seeds 20260723 --force
.\.venv\Scripts\python.exe -m roughness.scheme1_physics.cli train --config configs\scheme1_physics.yaml --models PW1,PE1,PW2,PE2 --folds 0 --seeds 20260723 --max-epochs 2 --batch-size 4
```

Expected:

- 四个run目录均为 `status=complete`；
- PW2父模型为PW1，PE2父模型为PE1；
- OOF无NaN；
- 门控值位于 `[0,1]`；
- 两个冻结审计均为 `unchanged=true`。

- [ ] **Step 6：清理微型运行与正式运行冲突**

不得删除原方案1文件。对 `outputs/scheme1_physics/` 中 `requested_max_epochs=2` 的run，正式命令依靠fingerprint不匹配自动重训；禁止手工将微型metadata改成200轮。

- [ ] **Step 7：启动正式完整运行**

Run:

```powershell
.\.venv\Scripts\python.exe -m roughness.scheme1_physics.cli run --config configs\scheme1_physics.yaml --max-epochs 200 --batch-size 4 --resume
```

Expected:

- PW0/PW1/PW2/PE0/PE1/PE2全部完成；
- 4个神经模型 × 5折 × 3种子 = 60个正式神经run；
- 正式总OOF共10,548行；
- `acceptance.json`给出唯一合法结论；
- 所有图表、Bootstrap和限制说明存在。

- [ ] **Step 8：运行正式产物审计**

Run:

```powershell
.\.venv\Scripts\python.exe -m roughness.scheme1_physics.cli evaluate --config configs\scheme1_physics.yaml
.\.venv\Scripts\python.exe -m pytest -q
```

并验证：

```powershell
Import-Csv outputs\scheme1_physics\oof_predictions.csv |
    Group-Object sample_id,model,seed |
    Where-Object Count -ne 1
```

Expected: PowerShell分组检查无输出；pytest全部PASS。

- [ ] **Step 9：最终人工科研审阅**

逐项检查：

- 半径选择是否跨折/种子不稳定；
- PE与PW差异是否达到1%＋折＋种子＋Bootstrap标准；
- PW2/PE2是否真正超过对应普通残差；
- 最佳候选是否超过M0；
- RMSE与 \(R^2\) 是否触发降级；
- 门控是否塌缩到近0或近1；
- 改善是否集中在单一工艺组合或少数组；
- 报告是否避免“真实半径识别”“真实物理概率”“独立因果贡献”等越界表述。

- [ ] **Step 10：任务检查点**

若Git以后可用，建议提交：

```powershell
git add src/roughness/scheme1_physics configs/scheme1_physics.yaml tests/scheme1_physics pyproject.toml
git commit -m "feat: complete gated physical roughness branch"
```

正式训练产物默认不加入Git，除非用户明确要求版本化实验输出。

---

## 实施顺序检查点

1. Tasks 1-3完成后，必须能在不训练神经网络的情况下生成双公式、选择尺度并构造无泄漏DataLoader。
2. Task 4完成后，必须通过普通残差和门控的代数恒等式测试。
3. Task 5完成后，必须先产出PW0/PE0及PW1/PE1。
4. Task 6只能在同折、同种子、同公式普通残差checkpoint存在后执行。
5. Tasks 7-9完成后，才能运行正式评估。
6. Task 10的正式训练只有在全套测试和两轮冒烟通过后启动。

## 完成定义

工程完成要求：

- 新增与原有测试全部通过；
- 原方案1关键产物哈希不变；
- 六模型三种子完整OOF共10,548行；
- 60个神经run均具有fingerprint、checkpoint、训练日志和完整metadata；
- 门控冻结审计全部通过；
- 评估、Bootstrap、图表与方法限制文件齐全。

科研完成要求：

- 每项比较均按预注册门槛给出通过或失败原因；
- 最终结论严格属于 `stable_effective`、`exploratory_increment` 或 `no_stable_increment`；
- 没有准确 \(r_e\) 时只报告有效尺度敏感性；
- 未满足门槛时不声称公式、残差或门控有效。
