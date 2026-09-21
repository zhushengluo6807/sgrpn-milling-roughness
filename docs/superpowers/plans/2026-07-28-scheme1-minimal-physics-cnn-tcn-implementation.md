# 方案1：最小物理—CNN-TCN 粗糙度预测实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task with review checkpoints.

**目标：** 在不补实验的前提下，基于现有 586 个区域片段和 212 组独立加工实验，完成无数据泄漏的方案1全流程；保留“1D-CNN → CNN-TCN → 工艺融合 → 残差 → 门控残差”网络主线，并对未知横向通道方向与端刃有效生成半径进行嵌套敏感性分析。

**架构：** 在现有 `src/roughness` 第一轮基线旁新增隔离的 `src/roughness/scheme1` 包。数据层复用既有 manifest 和固定外层折，按片段构建 1 秒窗口袋；特征层生成最小物理代理；模型层统一输出片段级 Ra；评估层以原始实验 `group_id` 为独立单位完成固定五折、三随机种子和配对 Bootstrap。所有预处理参数、方向假设、`r_e` 候选、早停和模型选择只能在当前外层训练折内确定。

**技术栈：** Python 3.11、NumPy、pandas、SciPy、scikit-learn、PyTorch、PyYAML、Matplotlib、pytest。项目命令使用 `.venv\Scripts\python.exe`；该虚拟环境的基础解释器来自 `python`。

**设计依据：** `docs/superpowers/specs/2026-07-27-two-route-roughness-model-design.md`

**版本控制说明：** 当前目录不是有效 Git 工作树。本计划不初始化仓库，也不伪造提交步骤。每个任务完成后运行列出的测试并记录检查点文件；若用户之后恢复 Git，再单独补提交。

---

## 一、不可变约束与交付定义

### 1. 固定数据约束

- 正式输入：`切削实验/预处理并切分后实验数据_不扩展剔除/segments`。
- 输入 CSV 已完成稳定段截取、中位数去偏置和 20–10000 Hz FFT 带通；方案1不得再次执行相同滤波或去中心。
- `Ch11_g=Z`；`Ch9_g/Ch10_g` 的 X/Y 方向未知。
- 采样率固定为 25,600 Hz。
- 每个片段一个 Ra 标签；窗口不是新的监督样本。
- 外层五折复用 `outputs/first_round_baseline/folds.csv`，划分单位为 `group_id`。
- 同一 `group_id` 的片段和窗口不得跨折。
- 片段权重为 `1/split_count`，使每个原始实验总权重为 1。
- 正式工艺基线 M0 为 `quadratic_process`，当前加权 MAE 为 `0.1046165189 μm`。

### 2. 网络主线

按以下顺序推进，不以手工特征模型替代：

```text
N1 1D-CNN
  ↓
N2 1D-CNN + TCN
  ↓
N3 工艺参数 + CNN-TCN
  ↓
N4 工艺基线 + CNN-TCN 残差
  ↓（仅在物理代理有效时）
N5 最小物理基线 + CNN-TCN 残差
  ↓（仅在普通残差有效时）
N6 门控残差
```

P1–P4 只承担解释、候选筛选和消融，不替代 N1–N6。

### 3. 完成交付物

最终必须产生：

```text
outputs/scheme1/
├── audit.json
├── window_index.csv
├── run_manifest.json
├── folds/
│   ├── fold_0_channel_stats.json
│   └── ...
├── features/
│   ├── segment_features.csv
│   └── feature_schema.json
├── classic/
│   ├── selection.csv
│   ├── oof_predictions.csv
│   └── metrics.csv
├── neural/
│   ├── checkpoints/<model>/fold_<k>/seed_<s>.pt
│   ├── logs/<model>/fold_<k>/seed_<s>.csv
│   ├── oof_predictions.csv
│   └── metrics.csv
└── evaluation/
    ├── summary_metrics.csv
    ├── paired_bootstrap.csv
    ├── acceptance.json
    └── figures/*.png
```

---

## 二、公共接口

### 1. 配置对象

文件：`src/roughness/scheme1/config.py`

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Scheme1Config:
    segments_dir: Path
    manifest_path: Path
    folds_path: Path
    output_dir: Path
    sample_rate_hz: int
    window_samples: int
    stride_samples: int
    re_candidates_mm: tuple[float, ...]
    seeds: tuple[int, ...]
    inner_splits: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
```

`configs/scheme1.yaml` 固定包含：

```yaml
sample_rate_hz: 25600
window_samples: 25600
stride_samples: 25600
re_candidates_mm: [0.075, 0.10, 0.20, 0.40, 0.80]
seeds: [20260723, 20260724, 20260725]
inner_splits: 4
welch_nperseg: 8192
welch_noverlap: 4096
nominal_band_min_halfwidth_hz: 5.0
nominal_band_relative_halfwidth: 0.05
hf_band_hz: [1000.0, 10000.0]
max_epochs: 200
patience: 20
learning_rate: 0.001
weight_decay: 0.0001
bootstrap_repetitions: 10000
```

### 2. 批次协议

文件：`src/roughness/scheme1/dataset.py`

所有神经模型接收同一种批次：

```python
{
    "signal": FloatTensor[B, W, 3, 25600],
    "window_mask": BoolTensor[B, W],
    "process": FloatTensor[B, 3],
    "physics": FloatTensor[B, P],
    "base_ra": FloatTensor[B],
    "target": FloatTensor[B],
    "sample_weight": FloatTensor[B],
    "segment_id": list[str],
    "group_id": list[str],
}
```

`W` 是当前批次最大窗口数。填充窗口必须全零，且不得参与池化。

### 3. 模型输出协议

文件：`src/roughness/scheme1/models.py`

```python
@dataclass
class ModelOutput:
    prediction: torch.Tensor
    residual: torch.Tensor | None = None
    gate: torch.Tensor | None = None
```

N1–N3 直接输出 `prediction`；N4–N5 输出 `base_ra + residual`；N6 输出 `base_ra + gate * residual`，其中 `gate=sigmoid(logit)`。

---

## 三、实施任务

## Task 1：建立方案1配置、目录和固定折校验

**文件：**

- 新建：`configs/scheme1.yaml`
- 新建：`src/roughness/scheme1/__init__.py`
- 新建：`src/roughness/scheme1/config.py`
- 新建：`src/roughness/scheme1/folds.py`
- 测试：`tests/scheme1/test_config.py`
- 测试：`tests/scheme1/test_folds.py`

**步骤 1：先写失败测试**

测试以下行为：

```python
def test_scheme1_config_has_frozen_protocol():
    cfg = load_scheme1_config("configs/scheme1.yaml")
    assert cfg.sample_rate_hz == 25_600
    assert cfg.window_samples == 25_600
    assert cfg.stride_samples == 25_600
    assert cfg.re_candidates_mm == (0.075, 0.10, 0.20, 0.40, 0.80)
    assert cfg.seeds == (20260723, 20260724, 20260725)

def test_outer_folds_match_manifest_groups():
    checked = validate_outer_folds(manifest, folds)
    assert checked["n_folds"] == 5
    assert checked["group_overlap_count"] == 0
```

还要覆盖：

- 路径解析均相对项目根目录，而非当前 shell 目录。
- `re_candidates_mm` 中若存在 `fz > 2*r_e` 的非法组合，后续几何函数必须显式拒绝。
- 每个 manifest 片段恰好出现在一个外层测试折。
- 同一 `group_id` 只能对应一个外层测试折。

**步骤 2：运行测试并确认失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1\test_config.py tests\scheme1\test_folds.py -q
```

**步骤 3：实现最小配置加载与折校验**

`validate_outer_folds()` 返回可序列化审计字典，并在重复片段、缺失片段、未知 group 或 group 泄漏时抛出 `ValueError`。

**步骤 4：再次运行测试**

预期：Task 1 测试全部通过。

**步骤 5：记录检查点**

将配置解析结果和折校验摘要写入 `outputs/scheme1/run_manifest.json`；不提交 Git。

---

## Task 2：实现信号文件审计与训练折通道统计

**文件：**

- 新建：`src/roughness/scheme1/signals.py`
- 测试：`tests/scheme1/test_signals.py`

**步骤 1：写失败测试**

覆盖：

```python
def test_load_signal_preserves_samples_and_channel_order(tmp_path):
    x = load_signal_csv(path)
    assert x.shape == (expected_rows, 3)
    assert list(x.columns) == ["Ch9_g", "Ch10_g", "Ch11_g"]

def test_channel_stats_use_training_segments_only():
    stats = fit_channel_stats(manifest, train_segment_ids={"a", "b"})
    assert stats.source_segment_ids == ("a", "b")

def test_standardization_is_global_not_per_window():
    scaled = apply_channel_stats(signal, stats)
    assert not np.allclose(scaled.mean(axis=0), 0.0)
```

还要拒绝：

- 缺列、重复列、NaN、Inf、空文件、常数通道、明显削顶比例超过审计阈值。
- CSV 行数与 manifest 样本数不一致。
- 外层测试片段进入统计拟合。

**步骤 2：运行测试并确认失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1\test_signals.py -q
```

**步骤 3：实现流式统计**

使用 Welford 算法对当前外层训练片段逐块计算每通道均值和标准差，避免将全部 CSV 同时读入内存。标准化只使用当前外层训练折统计，不进行逐窗归一化，不重复滤波。

核心接口：

```python
def audit_signal_files(manifest: pd.DataFrame) -> dict: ...
def fit_channel_stats(manifest: pd.DataFrame, train_segment_ids: set[str]) -> ChannelStats: ...
def load_signal_csv(path: Path) -> np.ndarray: ...
def standardize_signal(x: np.ndarray, stats: ChannelStats) -> np.ndarray: ...
```

**步骤 4：运行测试**

预期：通过，并生成 `audit.json` 与每折 `fold_k_channel_stats.json`。

---

## Task 3：构建无越界窗口索引和片段袋 Dataset

**文件：**

- 新建：`src/roughness/scheme1/windows.py`
- 新建：`src/roughness/scheme1/dataset.py`
- 测试：`tests/scheme1/test_windows.py`
- 测试：`tests/scheme1/test_dataset.py`

**步骤 1：写窗口失败测试**

精确规则：

```python
assert make_windows(25_600) == [(0, 25_600)]
assert make_windows(51_200) == [(0, 25_600), (25_600, 51_200)]
assert make_windows(30_000) == [(0, 25_600), (4_400, 30_000)]
```

- 长度不足 25,600 的片段必须在审计阶段报错，不能零填充成为正式窗口。
- 末端对齐窗口允许与前一窗口重叠，但只在存在不足 1 秒尾部时增加。
- 所有窗口的起止点必须位于同一片段。

**步骤 2：写 Dataset 失败测试**

验证：

- `__getitem__` 返回一个完整片段的所有窗口。
- `collate_segment_bags` 只在窗口维度填充。
- `window_mask` 与真实窗口数一致。
- 片段目标和权重只出现一次。
- A/B 映射只改变通道语义元数据，不改变原始数值顺序；对称模式对 Ch9/Ch10 置换不变。

**步骤 3：运行并确认失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1\test_windows.py tests\scheme1\test_dataset.py -q
```

**步骤 4：最小实现**

窗口索引列固定为：

```text
segment_id,group_id,csv_path,window_id,start_sample,end_sample,is_tail_aligned
```

Dataset 延迟加载 CSV；同一片段读取一次后切窗。正式训练 DataLoader 只在片段级打乱。

**步骤 5：运行测试并记录**

生成 `outputs/scheme1/window_index.csv`，审计其片段数等于 586、独立 group 数等于 212；若实际 manifest 不符则停止而不是硬编码通过。

---

## Task 4：实现工艺、频域与槽底几何代理特征

**文件：**

- 新建：`src/roughness/scheme1/features.py`
- 测试：`tests/scheme1/test_features.py`

**步骤 1：先写纯函数测试**

槽底精确公式：

```python
def bottom_geometry_ra_um(fz_mm: float, re_mm: float) -> float:
    if fz_mm > 2.0 * re_mm:
        raise ValueError("fz_mm must be <= 2 * re_mm")
    h_mm = re_mm - np.sqrt(re_mm**2 - (fz_mm / 2.0)**2)
    return 1000.0 * h_mm / 4.0
```

测试：

- 数值与手算结果一致。
- 小进给近似 `1000*fz²/(32*re)` 仅作校验，不作为正式值。
- `re=5 mm` 不在候选集合中。
- `re=0` 和非法根号域抛错。
- 输出有限、非负，并随 `fz` 增大、随 `re` 增大而减小。

**步骤 2：写信号特征测试**

用合成正弦和冲击信号验证：

- RMS、std、ptp、kurtosis、crest factor。
- Welch PSD 使用 `nperseg=8192, noverlap=4096`，短窗口时只允许 SciPy 明确降级并记录实际参数。
- 转频 `n/60`、三刃 TPF `3n/60` 及 2–4 倍频带能量。
- 频带半宽 `max(5 Hz, 0.05*中心频率)`。
- 高频能量范围 1000–10000 Hz。
- 频谱质心、归一化频谱熵。
- 带限位移代理排除 0 Hz，并满足全有限值。

位移代理只保存 `relative_unit` 标记，不写成 μm。

**步骤 3：写方向不变性测试**

定义三种表示：

- A：`Ch9=X, Ch10=Y`。
- B：`Ch9=Y, Ch10=X`。
- symmetric：横向幅值/能量的和、均值、最大值、最小值、二范数等。

对 Ch9/Ch10 交换后，symmetric 特征必须不变；A/B 方向专属字段必须对换。

**步骤 4：运行并确认失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1\test_features.py -q
```

**步骤 5：实现和导出**

特征按片段内所有窗口先计算，再以均值、标准差、最大值聚合为片段特征。特征缩放器只在当前外层训练折拟合。

输出：

- `segment_features.csv`
- `feature_schema.json`，记录单位、公式、通道语义、是否方向不变、是否物理代理。

不得覆盖既有第一轮 `ra_geo_um` 字段。若需要澄清历史字段，新增只读别名 `ra_geo_sidewall_proxy_um`，方案1代码禁止引用旧 `ra_geo_um` 作为槽底几何。

---

## Task 5：实现内层交叉拟合工艺基线与 P1–P4

**文件：**

- 新建：`src/roughness/scheme1/crossfit.py`
- 新建：`src/roughness/scheme1/classic.py`
- 测试：`tests/scheme1/test_crossfit.py`
- 测试：`tests/scheme1/test_classic.py`

**步骤 1：写交叉拟合失败测试**

核心接口：

```python
def make_group_inner_splits(
    frame: pd.DataFrame, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]: ...

def cross_fitted_predictions(
    estimator_factory, x, y, groups, sample_weight, splits
) -> np.ndarray: ...
```

测试保证：

- 每个外层训练样本得到且只得到一个 out-of-inner-fold 预测。
- 任一 inner validation group 不出现在对应 inner train。
- 输出顺序与输入 `segment_id` 一致。
- 外层测试标签从未传入候选选择函数。

**步骤 2：定义 P1–P4**

- M0：`quadratic_process`。
- P1：工艺 + Z 特征。
- P2：P1 + 横向对称特征。
- P3A/B：P2 + A 或 B 方向专属特征。
- P4：P1 或 P2 + 一个 `Ra_geo_bottom(re)` 候选；另含“无几何项”候选。

首版使用带标准化的加权 Ridge/ElasticNet 小型候选，不进行大规模搜索。所有 scaler、正则强度、映射 A/B 与 `r_e` 候选均在当前外层训练折的 group inner CV 内确定。

**步骤 3：定义选择对象**

```python
@dataclass(frozen=True)
class ClassicSelection:
    outer_fold: int
    xy_mode: Literal["A", "B", "symmetric"]
    re_mm: float | None
    model_name: str
    inner_weighted_mae: float
```

若 A 或 B 未在至少 3/5 外层折且三个种子总体上稳定胜出，最终解释采用 symmetric，不把选择结果称为真实安装方向。

`r_e` 是代理敏感性候选，报告中禁止称为实测半径。若各折不稳定或 P4 无增量，N5 不启用。

**步骤 4：运行测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1\test_crossfit.py tests\scheme1\test_classic.py -q
```

**步骤 5：运行经典模型**

```powershell
.\.venv\Scripts\python.exe -m roughness.scheme1.cli classic --config configs\scheme1.yaml
```

生成逐片段 OOF、逐折指标和候选选择表。检查 M0 与已有第一轮结果在数值容差内一致；若不一致，停止神经网络训练并定位原因。

---

## Task 6：实现 CNN、TCN 和掩码片段池化

**文件：**

- 新建：`src/roughness/scheme1/models.py`
- 测试：`tests/scheme1/test_models.py`

**步骤 1：写结构失败测试**

N1 编码器：

```text
Conv1d 3→16, kernel=31, stride=4
Conv1d 16→32, kernel=15, stride=4
Conv1d 32→64, kernel=9, stride=2
每层 GroupNorm + GELU
全局平均池化 + 最大池化
投影为 32 维窗口嵌入
```

验证参数量 `<1_000_000`。

N2 在 CNN 降采样后的序列上加入：

```text
64 channels, kernel=3, dilations=[1,2,4,8]
residual + GroupNorm + GELU + dropout=0.2
```

验证 CNN-TCN 总参数量 `<2_000_000`。

**步骤 2：写掩码测试**

```python
def test_masked_segment_pool_ignores_padding():
    out1 = model(real_windows, mask=[1, 1])
    out2 = model(pad(real_windows, 3), mask=[1, 1, 0, 0, 0])
    torch.testing.assert_close(out1.prediction, out2.prediction)
```

还要测试任意 batch size、不同窗口数、反向传播有限、输出 `[B]`。

**步骤 3：实现模型**

所有窗口先共享编码器，再对 32 维窗口嵌入做 masked mean pooling，最后产生一个片段输出。禁止把窗口摊平成独立 Ra 样本。

**步骤 4：运行测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1\test_models.py -q
```

---

## Task 7：实现 N3–N6 融合、残差和门控

**文件：**

- 修改：`src/roughness/scheme1/models.py`
- 测试：`tests/scheme1/test_fusion_models.py`

**步骤 1：写融合模型失败测试**

- N3 拼接标准化工艺参数 `[n,fz,ap]` 与 CNN-TCN 片段嵌入。
- N4 的 `base_ra` 必须来自 M0：外层训练样本用 inner cross-fitted 预测；外层测试样本用整个外层训练折拟合后的预测。
- N5 只在 P4 通过增量判断后使用相同规则生成最小物理基线。
- N6 `gate∈[0,1]`，且预测严格等于 `base_ra + gate*residual`。

用专门的 provenance 字段测试基线来源，防止把对训练样本的 in-sample 基线预测传入残差网络。

**步骤 2：实现工艺和物理分支**

工艺分支使用小型 MLP；物理分支只接收当前折已选择的代理项。模型不得学习或重新搜索 `r_e`，也不得从外层测试标签反推 A/B。

**步骤 3：实现复杂度开关**

CLI 在运行 N5 前读取 P4 判断；P4 无稳定增量时写入 `skipped_reason`。N6 仅在 N4 或 N5 普通残差相对对应基线有效后运行。

**步骤 4：运行测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1\test_fusion_models.py -q
```

---

## Task 8：实现训练器、早停和三种子固定五折

**文件：**

- 新建：`src/roughness/scheme1/training.py`
- 测试：`tests/scheme1/test_training.py`

**步骤 1：写训练器失败测试**

使用极小合成 Dataset 验证：

- 随机种子同时固定 Python、NumPy、PyTorch 和 DataLoader。
- train/validation 按 `group_id` 划分。
- 损失为片段加权 MAE 或配置指定的加权 Huber。
- 权重归一化不改变总体损失尺度。
- 只依据当前外层训练折内 validation weighted MAE 早停。
- patience=20，最多 200 epoch。
- 最优而非最后 epoch 权重被恢复。
- 日志记录 epoch、train loss、validation MAE、学习率和最佳 epoch。

**步骤 2：实现训练器**

核心接口：

```python
def train_one_fold(
    model_name: str,
    outer_fold: int,
    seed: int,
    config: Scheme1Config,
    manifest: pd.DataFrame,
) -> FoldRunResult: ...
```

每个 `(model, fold, seed)` 必须保存：

- 最佳 checkpoint。
- fold 内 channel stats、process scaler、physics scaler。
- inner baseline provenance。
- 训练日志。
- validation 和 outer-test 片段预测。

**步骤 3：实现可恢复执行**

若 checkpoint 与 run fingerprint（配置散列、manifest 散列、fold 文件散列、代码版本字段）一致则允许跳过；不一致必须重新训练，不能混用旧产物。

**步骤 4：运行测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1\test_training.py -q
```

**步骤 5：先做单折单种子 smoke**

```powershell
.\.venv\Scripts\python.exe -m roughness.scheme1.cli train --config configs\scheme1.yaml --models N1,N2 --folds 0 --seeds 20260723 --max-epochs 2
```

检查显存/内存、张量形状、日志、checkpoint 和 OOF 行数后，才允许全量训练。

---

## Task 9：实现指标、配对 Group Bootstrap 与验收门槛

**文件：**

- 新建：`src/roughness/scheme1/evaluation.py`
- 测试：`tests/scheme1/test_evaluation.py`

**步骤 1：写指标失败测试**

复用或包装现有 `roughness.metrics`，验证：

- weighted MAE、RMSE、R² 与手算一致。
- 聚合单位是片段，但 bootstrap 抽样单位是 `group_id`。
- 同一组内的全部片段在一次 bootstrap 中共同出现，并保留 `1/split_count` 权重。

**步骤 2：写配对 Bootstrap 测试**

```python
def paired_group_bootstrap(
    frame: pd.DataFrame,
    reference_col: str,
    candidate_col: str,
    repetitions: int = 10_000,
    seed: int = 20260723,
) -> BootstrapResult:
    ...
```

每次从 212 个 group 有放回抽样 212 次，计算同一抽样上的：

```text
delta_mae = MAE_reference - MAE_candidate
relative_improvement = delta_mae / MAE_reference
```

95% 分位区间下界 `>0` 才称为置信区间支持改善。

**步骤 3：编码逐级验收**

验收器必须输出机器可读原因：

- N2 vs N1：MAE 改善 `≥2%`，至少 `3/5` 折改善，种子平均同方向。
- N3 vs M0：MAE 改善 `≥3%`，至少 `3/5` 折改善。
- N4 vs M0：MAE 改善 `≥5%`，至少 `4/5` 折改善，三个种子分别都改善。
- N5 vs N4：进一步改善 `≥2%`，至少 `3/5` 折改善。
- N6 vs N5：进一步改善 `≥1%`，且折间 MAE 标准差不得明显增加；“明显”预注册为相对增加超过 20%。若 N5 因物理代理无效而未启用，则按预注册阶梯停止，不单独越级运行 N6。

方案1结论等级：

- 未发现稳定增量：N3/N4 对 M0 不足 3%，或折/种子方向反复。
- 探索性增量：最佳模型 `≥3%`、至少 `3/5` 折、三个种子平均方向一致。
- 稳定有效：最佳模型 `≥5%`、至少 `4/5` 折、三个种子分别改善、paired group bootstrap 95% 区间下界 `>0`。

稳定有效目标对应 MAE `<0.09939 μm`。

**步骤 4：运行测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scheme1\test_evaluation.py -q
```

---

## Task 10：实现 CLI、全链路审计和报告图

**文件：**

- 新建：`src/roughness/scheme1/cli.py`
- 新建：`src/roughness/scheme1/reporting.py`
- 修改：`pyproject.toml`
- 测试：`tests/scheme1/test_cli.py`
- 测试：`tests/scheme1/test_reporting.py`

**步骤 1：定义 CLI**

```text
python -m roughness.scheme1.cli audit
python -m roughness.scheme1.cli features
python -m roughness.scheme1.cli classic
python -m roughness.scheme1.cli train
python -m roughness.scheme1.cli evaluate
python -m roughness.scheme1.cli run
```

所有子命令要求 `--config configs/scheme1.yaml`。`run` 按 audit → features → classic → N1…N6 → evaluate 顺序执行，并遵守停止/降级规则。

**步骤 2：写 CLI 失败测试**

- `--help` 返回 0。
- 缺配置、折泄漏、数据审计失败返回非零。
- smoke 参数只影响临时运行，不修改正式 YAML。
- `evaluate` 在 OOF 不完整或重复时拒绝出报告。

**步骤 3：实现报告**

最少输出：

- 逐模型、逐折、逐种子 MAE/RMSE/R²。
- M0 与 N1–N6 配对比较。
- A/B/symmetric 与 `r_e` 候选内层选择频次。
- 预测—实测图、残差图。
- 按 n、fz、ap、版本和区域分组的误差图。
- 通道、网络、工艺、物理代理消融表。
- gate 分布（仅 N6）。
- 对物理代理和方向结论的限制说明。

**步骤 4：运行全部单元测试**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

预期：现有 16 项测试和新增方案1测试全部通过。

**步骤 5：运行全链路 smoke**

```powershell
.\.venv\Scripts\python.exe -m roughness.scheme1.cli run --config configs\scheme1.yaml --folds 0 --seeds 20260723 --max-epochs 2 --output-suffix smoke
```

确认所有产物可读取、OOF 无重复、group 无泄漏、预测均有限。

**步骤 6：运行正式经典模型与神经主线**

```powershell
.\.venv\Scripts\python.exe -m roughness.scheme1.cli run --config configs\scheme1.yaml
```

训练顺序和停止逻辑：

1. 先运行 M0、P1–P4、N1、N2。
2. N2 未过门槛时，后续融合继续使用表现更好的 CNN/CNN-TCN 编码器，但报告不得宣称 TCN 有增量。
3. 运行 N3、N4。
4. N3/N4 均未达到 3% 时停止新增 N5/N6 复杂度。
5. 仅 P4 有稳定增量时运行 N5。
6. 仅 N5 普通物理残差有效时运行 N6；N5 未启用时不越级运行 N6。

**步骤 7：记录最终检查点**

将完整命令、环境包版本、配置/manifest/folds 散列、开始结束时间写入 `run_manifest.json`。当前没有 Git 工作树，因此记录 `git_available=false`，不创建提交。

---

## 四、关键泄漏防护测试矩阵

以下测试必须在任何正式训练前通过：

| 风险 | 自动检查 |
|---|---|
| 同源片段跨外层折 | `group_id` train/test 交集为空 |
| 窗口跨片段 | 每窗起止点位于单一 segment |
| 窗口被当成独立标签 | Dataset 每片段只返回一个 target |
| 标准化偷看测试折 | scaler provenance 仅含 outer-train segment |
| A/B 选择偷看测试折 | selector API 不接收 outer-test y |
| `r_e` 选择偷看测试折 | 仅 inner grouped CV 产生 selection |
| 残差基线训练内拟合 | outer-train base_ra 全部来自 inner OOF |
| 早停偷看外层测试 | early stopper 只接收 inner validation metric |
| Bootstrap 把片段当独立组 | 抽样键固定为 `group_id` |
| 旧侧壁公式混入槽底 | 方案1源码不得引用 `ra_geo_um` |

新增守卫测试：

```powershell
Select-String -Path src\roughness\scheme1\*.py -Pattern "ra_geo_um" -SimpleMatch
```

预期无匹配。

---

## 五、性能与资源控制

- CSV 使用延迟加载；特征提取可缓存片段级结果。
- 每个 DataLoader batch 的单位是片段袋；先以保守 batch size 运行 smoke，再按显存调整。
- 混合精度只在数值一致性 smoke 通过后开启。
- 不做大规模超参数搜索，不因测试折结果改配置。
- 正式模型比较必须使用相同外层折、相同三个种子和相同片段权重。
- 若训练中断，只允许通过 fingerprint 一致的 checkpoint 恢复。

---

## 六、正式验收清单

实施完成不等于模型通过科研验收。最终分别检查：

### 工程完成

- 全部测试通过。
- audit、features、classic、train、evaluate 五阶段均可单独运行。
- 固定折、片段权重、标准化和基线 provenance 可审计。
- OOF 覆盖 586 个片段且无重复。
- 每折每种子 checkpoint、日志和预测齐全。
- 所有预测和中间物理量均有限。

### 科研结论

- M0 复现实验与当前 `0.1046165189 μm` 一致。
- N1–N6 按预注册阶梯比较，不跳级选择。
- 报告 fold、seed、group bootstrap，而不是只报总体均值。
- `r_e` 只称为有效生成尺度敏感性代理。
- A/B 只称为预测上受支持的方向假设；不作为传感器安装证明。
- 未标定位移代理只报告相对单位。
- 不把 CNN-TCN 有预测价值直接表述为完整物理机理已证实。

---

## 七、执行检查点

每完成一个任务，执行：

```powershell
.\.venv\Scripts\python.exe -m pytest <该任务测试文件> -q
```

每完成 Tasks 1–5、Tasks 6–8、Tasks 9–10 三个阶段，再执行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

若失败，按 `superpowers:systematic-debugging` 定位根因后再继续；在最终声称完成前，使用 `superpowers:verification-before-completion` 重新运行全套测试和 smoke，并以最新命令输出为证据。
