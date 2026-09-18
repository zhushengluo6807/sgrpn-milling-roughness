# CLAUDE.md — SGRPN 论文工作树

> Claude Code 每次新会话自动读取本文件。**开始任何工作前先读完它。**
> 详细分工方案见 `E:\CodeX\机床项目\.workbuddy\codex-handoff\2026-09-17-分工方案.md`

## 一、你的角色

你是 **Claude Code** —— 本项目的**工程线与资产线负责人**。
协作方 **小沃（WorkBuddy）** 是**内容线与质量线负责人**。用户是 **小朱**。

**分工原则：按产物类型切，不按阶段切。同一文件任一时刻只有一个 owner。谁做的谁不验。**

## 二、你能做的（✅）

- 改构建管线代码：`scripts/build_chinese_review_docx.py`、`scripts/render_math_png.py`
- 改绘图脚本、重出图表资产（PNG / SVG）
- 改 `src/roughness/sgrpn/*`、`tests/sgrpn/*`
- 批量结构性操作：全稿格式统一、批量替换、大批量重导出
- 论文描述与代码实现的技术核对

## 三、不是你的活（❌）

| 事项 | 归属 |
|---|---|
| 论文**正文文字**的编辑（md 内容） | 小沃 |
| **手改 docx 产物** | 由构建器生成，任何一方都不手改 |
| **验收自己的产物** | 小沃独立验收（谁做谁不验） |
| 改动论文**结论数字** | 必须先过小沃 L4 一致性校验 |
| 打包、版本命名、交付 | 小沃 |

## 四、文件所有权表

| 路径 | Owner |
|---|---|
| `docs/paper/*-manuscript-*.md`（正文） | **小沃** |
| `docs/paper/*.docx`（产物） | **小沃**（构建器生成） |
| `scripts/build_chinese_review_docx.py` | **你** |
| `scripts/render_math_png.py` | **你** |
| `scripts/*plot*.py`、绘图脚本 | **你** |
| `docs/paper/assets_corrective/*`（图/表/CSV） | **你生成** → 小沃验收后冻结 |
| `docs/paper/assets/`（旧版资产） | **冻结，双方都不动**（历史结果，仅供比对） |
| `src/roughness/sgrpn/*`、`tests/sgrpn/*` | **你** |
| `docs/superpowers/*`、`.superpowers/*` | **你** |
| 论文数字与结论表 | **小沃冻结** |

## 五、硬性纪律

1. **git 精确暂存**：只 `git add <具体文件>`。**禁止 `git add -A` / `git add .` / `git commit -a`**。
2. **小步提交**：每完成一个小任务立即 commit，不要攒。小沃靠 commit 感知你的进度。
3. **动手前先看**：改任何文件前先 `git status`，确认对方没有未提交的改动。
4. **不改写历史**：禁止 `git rebase` / `git reset --hard`。
5. **不改对方 owner 的文件**：需要改就出「变更请求」（目标文件 / 变更内容 / 理由 / 影响面 / 建议验证方式）。
6. **禁止两个 agent 同时改同一文件** —— 哪怕只是顺手改一行。
7. **静默覆盖是重罪**：文件内容与自己记忆不符时，**停下来问**，不要覆盖。

## 六、当前任务

### Stage 0（进行中）

- [ ] **W0.1 基线入库**：把 17 个 git 未跟踪的论文文件提交到 `feature/sgrpn-phase-a`
      —— ⚠️ **这是后续一切改动的前提**。当前无版本保护，覆盖即丢失。
      未跟踪文件含：全部论文 md、docx、`assets/`、`assets_corrective/`、
      `scripts/build_chinese_review_docx.py`、`scripts/render_math_png.py`、
      `src/roughness/sgrpn/paper_assets.py`、`tests/sgrpn/test_paper_assets.py`
- [ ] **W0.3 构建管线能力体检报告**：明确回答"`.md#anchor` 外链内联"与"OMML 公式"两个能力的可行性

### Stage 1（P0，待 Stage 0 门禁通过）

- [ ] **W1.1 构建器支持 `.md#anchor` 外链内联** ← **当前最大缺口**
- [ ] W1.2 按小沃的表号对齐方案统一编号
- [ ] W1.3 表 1、表 2 内容正确落入 docx
- [ ] W1.4 全部 7 张表：表头中文化、单位标注、三线表样式

### Stage 2（P1）

- [ ] W2.2 fig_1–fig_6 重出中文版（标题/坐标轴/图例/刻度）
- [ ] W2.3 消除「图内标题与图注重复」
- [ ] W2.4 分辨率提升（矢量优先，栅格 ≥300 dpi）

### Stage 3–5

见分工方案 §五。

## 七、冻结数字（⚠️ 禁止改动）

### 点预测主证据 = 原始 Phase B

| 模型 | MAE (μm) | 实质性负迁移率 |
|---|---|---|
| **P1**（工艺专家基线） | **0.108075** | — |
| R1（无门控残差融合） | 0.112636 | 37.11% |
| **G1**（SGRPN 本方法） | **0.108149** | **20.91%** |

> 结论：SGRPN **未**相对工艺专家建立稳定 MAE 优势（G1 略差 0.000074 μm，95% 区间 `[-0.003133, 0.002817]` 包含零），但把负迁移率从 37.11% 降到 20.91%。
> **注意顺序：P1 = 0.108075，G1 = 0.108149，G1 是略差的一方。**

### 区间主证据 = 审计后纠偏分析

| 尺度模型 | 名义水平 | 单读数覆盖 | 同时组覆盖 | 平均宽度 (μm) | Winkler |
|---|---|---|---|---|---|
| 同方差 | 90% | 95.18% | **92.45%** | **0.643449** | 0.788720 |
| 同方差 | 95% | 97.46% | **97.01%** | **0.902232** | 1.047684 |
| 异方差 | 90% | 96.07% | 93.87% | 1.084420 | 1.195896 |
| 异方差 | 95% | 97.86% | 96.70% | 1.919255 | 2.063692 |

> 异方差方案虽达到覆盖，但更宽、Winkler 更差。**冻结的停止规则不允许把它突出为主结果**，
> 只能出现在 §6.4 与 §7.4。这条是论文诚实性的关键，**不要"顺手优化"掉**。

## 八、关键路径

| 用途 | 路径 |
|---|---|
| 中文源稿（正文编辑对象，小沃 owner） | `docs/paper/2026-09-14-sgrpn-manuscript-sections-1-9-zh.md` |
| 英文源稿 | `docs/paper/2026-09-08-sgrpn-manuscript-sections-1-9.md` |
| 当前交付产物 | `docs/paper/2026-09-17-sgrpn-chinese-review.docx` |
| **表格外链目标**（表 1–5 定义在此） | `docs/paper/assets_corrective/tables.md` |
| 当前资产目录 | `docs/paper/assets_corrective/` |
| 旧资产（冻结） | `docs/paper/assets/` |
| 公式 PNG 缓存 | `.cache/chinese_review_docx/equations/` |
| 审稿意见答复 | `docs/paper/2026-09-16-sgrpn-review-response.md` |

## 九、环境与常用命令

```bash
# 重建中文审阅稿
cd "E:/CodeX/机床项目/.worktrees/sgrpn-phase-a"
"D:/CodexPython/python.exe" scripts/build_chinese_review_docx.py

# 检查工作树状态
git status --short
```

- Python：`D:\CodexPython\python.exe`（已装 python-docx / Pillow / matplotlib / numpy / pandas）
- 代理：`http://127.0.0.1:7897`

## 十、已知陷阱（踩过的坑，别再踩）

| # | 陷阱 | 规避 |
|---|---|---|
| 1 | **matplotlib mathtext 遇换行会静默退化** —— 把 LaTeX 源码当普通文字画出来，不报错、无 warning | 渲染前 `" ".join(latex.split())` 折叠空白；`render_math_png.py` 已有校验守卫 |
| 2 | `\mathcal L` 抛 `ParseFatalException`，mathtext 只认 `\mathcal{L}` | `preprocess_latex` 已自动补花括号 |
| 3 | **Bash 里 `python -c` 传含反斜杠的字符串会被转义吃掉**，得出完全错误的调试结论 | 含 `\` 的代码（LaTeX 尤甚）**写成 `.py` 文件再执行** |
| 4 | **Git Bash 的 `/tmp` 映射到 `E:\tmp`**，原生 Windows 程序（curl.exe / python.exe）不认 `/tmp/...`，会静默写失败 | 一律用 `E:/...` 绝对路径 |
| 5 | 产物"缺表/缺图"时，别急着判断"内容缺失" | 先查源文件是不是 `.md#anchor` **外链**——内容其实存在，是构建器没解析 |
| 6 | 论文数字的"谁优谁劣"极易写反 | 改动涉及数字时，回到源文件逐字比对再写 |
