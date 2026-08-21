# 首轮工艺参数基线训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在独立CUDA虚拟环境中构建586片段、212实验组的无泄漏数据清单，并完成可复现的5折工艺参数与几何先验基线训练。

**Architecture:** 使用配置文件集中保存输入路径和随机种子；数据清单模块只负责Excel连接、CSV定位和派生字段；划分模块只负责实验组级折定义；基线模块只消费清单与固定折；命令行流水线把审计、训练和输出串联。所有模型共享同一折定义，预处理器只在训练折拟合。

**Tech Stack:** Python 3.12、PyTorch 2.11 CUDA 12.8、pandas、NumPy、SciPy、scikit-learn、openpyxl、PyYAML、Matplotlib、Seaborn、pytest。

## Global Constraints

- 基础解释器固定为 `D:\CodexPython\python.exe`，虚拟环境固定为 `E:\CodeX\机床项目\.venv`。
- 振动根目录固定为 `E:\CodeX\机床项目\切削实验\预处理并切分后实验数据_不扩展剔除\segments`。
- 标签表固定为 `E:\CodeX\机床项目\切削实验\粗糙度测量记录表_按切分片段.xlsx`。
- v3/v4工艺参数分别来自 `实验记录表_改进版_v3.xlsx` 和 `实验记录表_改进版_v4.xlsx`。
- 连接键固定为 `(版本, 执行顺序)`；`n/fz/ap` 固定读取 G/H/J 列对应字段。
- 随机种子固定为 `20260723`；折数固定为5。
- 同一 `group_id` 不得跨折；每片段权重固定为 `1 / split_count`。
- 首轮不训练CNN/TCN，不做超参数搜索，不使用窗口级随机划分。
- 当前工作区的 `.git` 目录为空且不是有效仓库；不得擅自初始化Git，计划中的提交检查点只记录文件和测试结果。

---

## File Map

- Create: `requirements/torch-cu128.txt` — CUDA版PyTorch安装入口。
- Create: `requirements/base.txt` — 首轮公共依赖。
- Create: `pyproject.toml` — 包与pytest配置。
- Create: `configs/first_round.yaml` — 数据路径、折数、种子和输出路径。
- Create: `src/roughness/config.py` — 配置加载和路径校验。
- Create: `src/roughness/manifest.py` — Excel连接、CSV解析、派生字段和审计。
- Create: `src/roughness/splits.py` — 确定性的组级5折。
- Create: `src/roughness/metrics.py` — 普通与组均衡回归指标。
- Create: `src/roughness/baselines.py` — 固定基线模型和折外预测。
- Create: `src/roughness/reporting.py` — 汇总表和诊断图。
- Create: `src/roughness/cli.py` — 端到端命令行入口。
- Create: `tests/test_config.py`、`tests/test_manifest.py`、`tests/test_splits.py`、`tests/test_metrics.py`、`tests/test_baselines.py`。

---

### Task 1: 创建并验证独立训练环境

**Files:**
- Create: `requirements/torch-cu128.txt`
- Create: `requirements/base.txt`
- Create: `pyproject.toml`
- Create: `src/roughness/__init__.py`

**Interfaces:**
- Consumes: `D:\CodexPython\python.exe`、NVIDIA驱动572.47、CUDA 12.8兼容驱动。
- Produces: `.venv\Scripts\python.exe`、可用CUDA PyTorch、可导入的 `roughness` 包。

- [ ] **Step 1: 写依赖文件**

`requirements/torch-cu128.txt`：

```text
--index-url https://download.pytorch.org/whl/cu128
torch==2.11.0
```

`requirements/base.txt`：

```text
numpy>=2.0,<3
pandas>=2.2,<3
scipy>=1.14,<2
scikit-learn>=1.6,<2
openpyxl>=3.1,<4
matplotlib>=3.9,<4
seaborn>=0.13,<1
tqdm>=4.66,<5
PyYAML>=6.0,<7
tensorboard>=2.18,<3
joblib>=1.4,<2
pytest>=8.3,<10
```

`pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "roughness-training"
version = "0.1.0"
requires-python = ">=3.12,<3.13"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

`src/roughness/__init__.py`：

```python
__version__ = "0.1.0"
```

- [ ] **Step 2: 创建虚拟环境并升级打包工具**

Run:

```powershell
& 'D:\CodexPython\python.exe' -m venv 'E:\CodeX\机床项目\.venv'
& 'E:\CodeX\机床项目\.venv\Scripts\python.exe' -m pip install --upgrade pip setuptools wheel
```

Expected: 两条命令退出码均为0，且 `.venv\Scripts\python.exe` 存在。

- [ ] **Step 3: 安装PyTorch和公共依赖**

Run:

```powershell
& '.venv\Scripts\python.exe' -m pip install -r requirements\torch-cu128.txt
& '.venv\Scripts\python.exe' -m pip install -r requirements\base.txt
& '.venv\Scripts\python.exe' -m pip install -e .
```

Expected: 三条命令退出码均为0，无依赖冲突。

- [ ] **Step 4: 验证GPU和依赖**

Run:

```powershell
& '.venv\Scripts\python.exe' -c "import torch, sklearn, pandas; x=torch.ones(1024,device='cuda'); print(torch.__version__); print(torch.cuda.get_device_name(0)); print(float(x.sum()))"
```

Expected: 输出PyTorch版本、`NVIDIA GeForce RTX 4060 Laptop GPU` 和 `1024.0`。

- [ ] **Step 5: 锁定实际依赖版本**

Run:

```powershell
& '.venv\Scripts\python.exe' -m pip freeze | Set-Content -Encoding UTF8 requirements\locked.txt
```

Expected: `requirements/locked.txt` 非空且包含 `torch==2.11.0`、`scikit-learn` 和 `pandas`。

- [ ] **Step 6: 记录检查点**

记录新增文件及Task 1验证输出；当前不是有效Git仓库，不执行 `git commit`。

---

### Task 2: 配置加载与数据路径校验

**Files:**
- Create: `configs/first_round.yaml`
- Create: `src/roughness/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `TrainingConfig`；`load_config(path: Path) -> TrainingConfig`。
- Consumers: Task 3和Task 6。

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path
import pytest
from roughness.config import load_config

def test_load_config_resolves_existing_paths(tmp_path: Path):
    for name in ["labels.xlsx", "v3.xlsx", "v4.xlsx"]:
        (tmp_path / name).touch()
    (tmp_path / "segments").mkdir()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "labels: labels.xlsx\nv3_records: v3.xlsx\nv4_records: v4.xlsx\n"
        "segments_root: segments\noutput_dir: out\nseed: 20260723\nn_splits: 5\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.labels == tmp_path / "labels.xlsx"
    assert cfg.seed == 20260723
    assert cfg.n_splits == 5

def test_load_config_rejects_missing_input(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("labels: missing.xlsx\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required config keys|Input path does not exist"):
        load_config(cfg_path)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v`

Expected: FAIL，原因是 `roughness.config` 尚不存在。

- [ ] **Step 3: 写最小实现**

```python
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class TrainingConfig:
    labels: Path
    v3_records: Path
    v4_records: Path
    segments_root: Path
    output_dir: Path
    seed: int
    n_splits: int

def load_config(path: Path) -> TrainingConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {"labels", "v3_records", "v4_records", "segments_root", "output_dir", "seed", "n_splits"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    base = path.parent
    def resolved(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (base / p).resolve()
    inputs = {key: resolved(raw[key]) for key in ["labels", "v3_records", "v4_records", "segments_root"]}
    for key, input_path in inputs.items():
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {key}={input_path}")
    return TrainingConfig(
        **inputs,
        output_dir=resolved(raw["output_dir"]),
        seed=int(raw["seed"]),
        n_splits=int(raw["n_splits"]),
    )
```

`configs/first_round.yaml`：

```yaml
labels: ../切削实验/粗糙度测量记录表_按切分片段.xlsx
v3_records: ../切削实验/实验记录表_改进版_v3.xlsx
v4_records: ../切削实验/实验记录表_改进版_v4.xlsx
segments_root: ../切削实验/预处理并切分后实验数据_不扩展剔除/segments
output_dir: ../outputs/first_round_baseline
seed: 20260723
n_splits: 5
```

- [ ] **Step 4: 运行测试并确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v`

Expected: 2 passed。

- [ ] **Step 5: 记录检查点**

记录Task 2文件和测试输出；不初始化Git。

---

### Task 3: 构建586片段统一清单

**Files:**
- Create: `src/roughness/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `TrainingConfig`。
- Produces: `build_manifest(config) -> tuple[pd.DataFrame, dict]`。
- Manifest required columns: `sample_id, group_id, signal_path, ra_mean, ra_std, ra_range, n_rpm, fz_mm_per_tooth, ap_mm, ra_geo_um, sample_weight`。

- [ ] **Step 1: 写失败测试，覆盖路径替换和参数连接**

```python
from pathlib import Path, PureWindowsPath
import pytest
from roughness.manifest import relative_after_segments, geometry_ra_um

def test_relative_after_segments_preserves_nested_suffix():
    source = r"切削实验\预处理并切分后实验数据\segments\500\3_seg1.csv"
    assert relative_after_segments(source) == Path("500") / "3_seg1.csv"

def test_geometry_ra_converts_mm_to_um():
    assert geometry_ra_um(0.15, radius_mm=5.0) == pytest.approx(0.140625)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_manifest.py -v`

Expected: FAIL，原因是函数尚不存在。

- [ ] **Step 3: 实现路径、表头规范化和连接函数**

```python
from pathlib import Path, PureWindowsPath
import numpy as np
import pandas as pd
from .config import TrainingConfig

def relative_after_segments(raw: str) -> Path:
    parts = PureWindowsPath(str(raw)).parts
    lowered = [p.lower() for p in parts]
    if "segments" not in lowered:
        raise ValueError(f"Path does not contain segments: {raw}")
    index = lowered.index("segments")
    suffix = parts[index + 1:]
    if not suffix:
        raise ValueError(f"Path has no suffix after segments: {raw}")
    return Path(*suffix)

def geometry_ra_um(fz_mm_per_tooth: float, radius_mm: float = 5.0) -> float:
    return float((fz_mm_per_tooth ** 2 / (32.0 * radius_mm)) * 1000.0)

def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).replace("\n", "").strip() for c in frame.columns]
    return frame

def _load_process_table(path: Path, version: str) -> pd.DataFrame:
    frame = _normalize_columns(pd.read_excel(path, sheet_name="实验记录表", header=2))
    frame = frame.rename(columns={"执行顺序": "execution_order", "转速(rpm)": "n_rpm", "进给(mm/z)": "fz_mm_per_tooth", "切深(mm)": "ap_mm"})
    frame = frame[["execution_order", "n_rpm", "fz_mm_per_tooth", "ap_mm"]].dropna(subset=["execution_order"])
    frame["execution_order"] = frame["execution_order"].astype(int)
    frame["version"] = version
    if frame.duplicated(["version", "execution_order"]).any():
        raise ValueError(f"Duplicate process keys in {path}")
    return frame
```

- [ ] **Step 4: 实现完整清单与硬审计**

```python
def build_manifest(config: TrainingConfig) -> tuple[pd.DataFrame, dict]:
    labels = _normalize_columns(pd.read_excel(config.labels, sheet_name="粗糙度测量记录", header=3))
    labels = labels.rename(columns={
        "版本": "version", "执行顺序": "execution_order", "原始实验组": "group_id",
        "该组切分数": "split_count", "区域编号": "region_index", "片段时长(s)": "duration_s",
        "对应信号文件": "source_signal_path", "Ra第1次(μm)": "ra_1", "Ra第2次(μm)": "ra_2",
        "Ra第3次(μm)": "ra_3", "Ra平均值(μm)": "ra_mean",
    })
    labels = labels.dropna(subset=["version", "execution_order", "region_index"]).copy()
    labels["execution_order"] = labels["execution_order"].astype(int)
    labels["region_index"] = labels["region_index"].astype(int)
    labels["split_count"] = labels["split_count"].astype(int)
    process = pd.concat([
        _load_process_table(config.v3_records, "v3"),
        _load_process_table(config.v4_records, "v4"),
    ], ignore_index=True)
    manifest = labels.merge(process, on=["version", "execution_order"], how="left", validate="many_to_one")
    manifest["sample_id"] = manifest.apply(lambda r: f"{r.version}_{r.execution_order}_seg{r.region_index}", axis=1)
    manifest["signal_path"] = manifest["source_signal_path"].map(lambda p: config.segments_root / relative_after_segments(p))
    readings = manifest[["ra_1", "ra_2", "ra_3"]].astype(float)
    manifest["ra_std"] = readings.std(axis=1, ddof=1)
    manifest["ra_range"] = readings.max(axis=1) - readings.min(axis=1)
    manifest["ra_geo_um"] = manifest["fz_mm_per_tooth"].map(geometry_ra_um)
    manifest["sample_weight"] = 1.0 / manifest["split_count"]
    required = ["sample_id", "group_id", "ra_mean", "n_rpm", "fz_mm_per_tooth", "ap_mm", "signal_path"]
    if len(manifest) != 586 or manifest["group_id"].nunique() != 212:
        raise ValueError(f"Unexpected dataset shape: rows={len(manifest)}, groups={manifest['group_id'].nunique()}")
    if manifest[required].isna().any().any():
        raise ValueError("Required manifest values contain missing data")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id")
    missing_files = [str(p) for p in manifest["signal_path"] if not Path(p).is_file()]
    if missing_files:
        raise ValueError(f"Missing signal files: {missing_files[:10]}")
    group_weights = manifest.groupby("group_id")["sample_weight"].sum()
    if not np.allclose(group_weights.to_numpy(), 1.0):
        raise ValueError("Group weights do not sum to one")
    audit = {
        "rows": len(manifest), "groups": int(manifest["group_id"].nunique()),
        "missing_signal_files": 0, "n_range": [float(manifest.n_rpm.min()), float(manifest.n_rpm.max())],
        "fz_range": [float(manifest.fz_mm_per_tooth.min()), float(manifest.fz_mm_per_tooth.max())],
        "ap_range": [float(manifest.ap_mm.min()), float(manifest.ap_mm.max())],
    }
    return manifest, audit
```

- [ ] **Step 5: 增加真实数据集成测试**

```python
def test_real_manifest_has_expected_shape():
    cfg = load_config(Path("configs/first_round.yaml"))
    manifest, audit = build_manifest(cfg)
    assert len(manifest) == 586
    assert manifest.group_id.nunique() == 212
    assert audit["missing_signal_files"] == 0
    assert audit["n_range"] == [4000.0, 8500.0]
    assert audit["fz_range"] == [0.03, 0.15]
    assert audit["ap_range"] == [0.5, 2.0]
```

- [ ] **Step 6: 运行Task 3测试**

Run: `.venv\Scripts\python.exe -m pytest tests/test_manifest.py -v`

Expected: 全部PASS，并实际验证586个CSV存在。

- [ ] **Step 7: 记录检查点**

记录清单审计输出和测试结果；不初始化Git。

---

### Task 4: 固定实验组级5折

**Files:**
- Create: `src/roughness/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Produces: `make_group_folds(manifest, n_splits, seed) -> pd.DataFrame`，列为 `sample_id, group_id, fold`。
- Consumers: Task 6基线训练。

- [ ] **Step 1: 写失败测试**

```python
import pandas as pd
from roughness.splits import make_group_folds

def test_group_folds_are_deterministic_and_leak_free():
    manifest = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(12)],
        "group_id": [f"g{i//2}" for i in range(12)],
    })
    a = make_group_folds(manifest, n_splits=3, seed=7)
    b = make_group_folds(manifest, n_splits=3, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby("group_id").fold.nunique().max() == 1
    assert a.sample_id.nunique() == len(manifest)
```

- [ ] **Step 2: 运行并确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_splits.py -v`

Expected: FAIL，函数尚不存在。

- [ ] **Step 3: 实现确定性分组折**

```python
import numpy as np
import pandas as pd

def make_group_folds(manifest: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    groups = np.array(sorted(manifest["group_id"].astype(str).unique()))
    if len(groups) < n_splits:
        raise ValueError("Number of groups is smaller than n_splits")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    mapping = {group: fold for fold, chunk in enumerate(np.array_split(groups, n_splits)) for group in chunk}
    result = manifest[["sample_id", "group_id"]].copy()
    result["fold"] = result["group_id"].astype(str).map(mapping).astype(int)
    if result.groupby("group_id")["fold"].nunique().max() != 1:
        raise AssertionError("Group leakage detected")
    return result.sort_values("sample_id").reset_index(drop=True)
```

- [ ] **Step 4: 运行并确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_splits.py -v`

Expected: 1 passed。

- [ ] **Step 5: 记录检查点**

记录折数量、每折组数和测试结果。

---

### Task 5: 指标与固定基线模型

**Files:**
- Create: `src/roughness/metrics.py`
- Create: `src/roughness/baselines.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Produces: `regression_metrics(y_true, y_pred, sample_weight=None) -> dict[str, float]`。
- Produces: `model_factories(seed) -> dict[str, callable]`。
- Produces: `cross_validated_baselines(manifest, folds, seed) -> tuple[pd.DataFrame, pd.DataFrame]`。

- [ ] **Step 1: 写指标失败测试**

```python
import numpy as np
from roughness.metrics import regression_metrics

def test_regression_metrics_exact_prediction():
    y = np.array([1.0, 2.0, 3.0])
    result = regression_metrics(y, y, np.array([0.5, 0.25, 0.25]))
    assert result == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}
```

- [ ] **Step 2: 实现指标**

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(y_true, y_pred, sample_weight=None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)),
        "rmse": float(mean_squared_error(y_true, y_pred, sample_weight=sample_weight) ** 0.5),
        "r2": float(r2_score(y_true, y_pred, sample_weight=sample_weight)),
    }
```

- [ ] **Step 3: 写基线失败测试**

```python
from roughness.baselines import model_factories

def test_expected_baselines_exist():
    names = set(model_factories(20260723))
    assert names == {"dummy_mean", "linear_process", "quadratic_process", "hist_process", "calibrated_geometry", "process_plus_geometry"}
```

- [ ] **Step 4: 实现固定模型工厂**

```python
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def model_factories(seed: int):
    return {
        "dummy_mean": lambda: DummyRegressor(strategy="mean"),
        "linear_process": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "quadratic_process": lambda: make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=1.0)),
        "hist_process": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed),
        "calibrated_geometry": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "process_plus_geometry": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed),
    }
```

- [ ] **Step 5: 实现折外训练循环**

```python
import time
import pandas as pd
from .metrics import regression_metrics

FEATURES = {
    "dummy_mean": ["n_rpm"],
    "linear_process": ["n_rpm", "fz_mm_per_tooth", "ap_mm"],
    "quadratic_process": ["n_rpm", "fz_mm_per_tooth", "ap_mm"],
    "hist_process": ["n_rpm", "fz_mm_per_tooth", "ap_mm"],
    "calibrated_geometry": ["ra_geo_um"],
    "process_plus_geometry": ["n_rpm", "fz_mm_per_tooth", "ap_mm", "ra_geo_um"],
}

def _fit_with_weight(model, x, y, weight):
    if hasattr(model, "steps"):
        final_name = model.steps[-1][0]
        model.fit(x, y, **{f"{final_name}__sample_weight": weight})
    else:
        model.fit(x, y, sample_weight=weight)

def cross_validated_baselines(manifest: pd.DataFrame, folds: pd.DataFrame, seed: int):
    data = manifest.merge(folds[["sample_id", "fold"]], on="sample_id", validate="one_to_one")
    predictions = []
    fold_metrics = []
    for model_name, factory in model_factories(seed).items():
        features = FEATURES[model_name]
        for fold in sorted(data.fold.unique()):
            train = data[data.fold != fold]
            valid = data[data.fold == fold]
            model = factory()
            fit_start = time.perf_counter()
            _fit_with_weight(model, train[features], train.ra_mean, train.sample_weight)
            fit_seconds = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            pred = model.predict(valid[features])
            predict_seconds = time.perf_counter() - predict_start
            segment_metrics = regression_metrics(valid.ra_mean, pred)
            weighted_metrics = {f"weighted_{key}": value for key, value in regression_metrics(valid.ra_mean, pred, valid.sample_weight).items()}
            fold_metrics.append({"model": model_name, "fold": int(fold), **segment_metrics, **weighted_metrics, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds})
            predictions.extend({"sample_id": sid, "model": model_name, "fold": int(fold), "y_true": float(y), "y_pred": float(p)} for sid, y, p in zip(valid.sample_id, valid.ra_mean, pred))
    prediction_frame = pd.DataFrame(predictions)
    metric_frame = pd.DataFrame(fold_metrics)
    expected = len(data) * len(FEATURES)
    if len(prediction_frame) != expected or prediction_frame.duplicated(["sample_id", "model"]).any():
        raise AssertionError("Incomplete or duplicate out-of-fold predictions")
    return prediction_frame, metric_frame
```

- [ ] **Step 6: 运行Task 5测试**

Run: `.venv\Scripts\python.exe -m pytest tests/test_metrics.py tests/test_baselines.py -v`

Expected: 全部PASS。

- [ ] **Step 7: 记录检查点**

记录模型清单和测试输出。

---

### Task 6: 端到端首轮训练命令

**Files:**
- Create: `src/roughness/cli.py`
- Create: `src/roughness/reporting.py`
- Modify: `pyproject.toml`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 2–5全部接口。
- Produces: `roughness-first-round --config configs/first_round.yaml`。

- [ ] **Step 1: 写CLI集成测试**

```python
from pathlib import Path
from roughness.cli import run

def test_run_writes_required_outputs(tmp_path: Path, monkeypatch):
    # 使用真实配置但将输出目录覆盖到tmp_path；训练仅使用小型固定模型。
    result = run(Path("configs/first_round.yaml"), output_override=tmp_path)
    for name in ["manifest.csv", "audit.json", "folds.csv", "oof_predictions.csv", "fold_metrics.csv", "summary_metrics.csv", "run_metadata.json", "prediction_scatter.png", "residual_plot.png"]:
        assert (tmp_path / name).is_file()
    assert result["rows"] == 586
```

- [ ] **Step 2: 实现汇总和诊断图**

```python
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["mae", "rmse", "r2", "weighted_mae", "weighted_rmse", "weighted_r2", "fit_seconds", "predict_seconds"]
    summary = fold_metrics.groupby("model")[columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()

def save_diagnostic_plots(predictions: pd.DataFrame, summary: pd.DataFrame, output: Path) -> str:
    best_model = summary.sort_values("weighted_mae_mean").iloc[0]["model"]
    chosen = predictions[predictions.model == best_model]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(chosen.y_true, chosen.y_pred, s=18, alpha=0.7)
    limits = [min(chosen.y_true.min(), chosen.y_pred.min()), max(chosen.y_true.max(), chosen.y_pred.max())]
    ax.plot(limits, limits, "k--", linewidth=1)
    ax.set(xlabel="Measured Ra (um)", ylabel="Predicted Ra (um)", title=f"OOF Prediction: {best_model}")
    fig.tight_layout(); fig.savefig(output / "prediction_scatter.png", dpi=180); plt.close(fig)
    residual = chosen.y_pred - chosen.y_true
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(chosen.y_pred, residual, s=18, alpha=0.7); ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Predicted Ra (um)", ylabel="Residual (um)", title=f"OOF Residuals: {best_model}")
    fig.tight_layout(); fig.savefig(output / "residual_plot.png", dpi=180); plt.close(fig)
    return str(best_model)
```

- [ ] **Step 3: 实现CLI与输出**

```python
import argparse, json, platform, sys, time
from pathlib import Path
import pandas as pd
import sklearn, torch
from .config import load_config
from .manifest import build_manifest
from .splits import make_group_folds
from .baselines import cross_validated_baselines
from .reporting import summarize_metrics, save_diagnostic_plots

def run(config_path: Path, output_override: Path | None = None) -> dict:
    cfg = load_config(config_path)
    output = output_override or cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest, audit = build_manifest(cfg)
    folds = make_group_folds(manifest, cfg.n_splits, cfg.seed)
    predictions, fold_metrics = cross_validated_baselines(manifest, folds, cfg.seed)
    summary = summarize_metrics(fold_metrics)
    best_model = save_diagnostic_plots(predictions, summary, output)
    manifest.to_csv(output / "manifest.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(output / "folds.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {"seed": cfg.seed, "n_splits": cfg.n_splits, "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "sklearn": sklearn.__version__, "cuda_available": torch.cuda.is_available(), "best_model_by_weighted_mae": best_model}
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)

if __name__ == "__main__":
    main()
```

在 `pyproject.toml` 添加：

```toml
[project.scripts]
roughness-first-round = "roughness.cli:main"
```

- [ ] **Step 4: 运行完整测试套件**

Run: `.venv\Scripts\python.exe -m pytest -v`

Expected: 全部测试PASS，无warning导致的失败。

- [ ] **Step 5: 执行真实首轮训练**

Run:

```powershell
& '.venv\Scripts\roughness-first-round.exe' --config 'configs\first_round.yaml'
```

Expected: 退出码0，并在 `outputs/first_round_baseline` 生成七个数据/元数据文件和两个诊断图。

- [ ] **Step 6: 验证验收条件**

Run:

```powershell
& '.venv\Scripts\python.exe' -c "import pandas as pd; m=pd.read_csv('outputs/first_round_baseline/manifest.csv'); f=pd.read_csv('outputs/first_round_baseline/folds.csv'); s=pd.read_csv('outputs/first_round_baseline/summary_metrics.csv'); assert len(m)==586 and m.group_id.nunique()==212; assert f.groupby('group_id').fold.nunique().max()==1; print(s.to_string(index=False))"
```

Expected: 断言通过，工艺参数最佳模型显著优于均值基线；若MAE不在0.10–0.12 μm或R²不在0.78–0.85，保留结果并进入数据/指标诊断，不修改折划分追历史数值。

- [ ] **Step 7: 记录最终检查点**

记录测试总数、GPU验证、586/212审计、各模型指标和输出路径；当前不是有效Git仓库，不执行提交。

---

## Plan Self-Review

- Spec coverage: 环境、586/212清单、第二套CSV、v3/v4映射、组级5折、组均衡权重、六个基线、指标和验收均有对应Task。
- Placeholder scan: 无TBD、TODO、“稍后实现”或未定义接口。
- Type consistency: `TrainingConfig`、`build_manifest`、`make_group_folds`、`cross_validated_baselines` 和 `run` 的输入输出在上下游一致。
- Scope: 仅实现首轮工艺参数/几何基线，不提前加入深度网络。
