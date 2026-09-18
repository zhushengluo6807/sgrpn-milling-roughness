# SGRPN 独立审稿报告

**审查日期：2026-09-10**  
**审查方式：独立 Agent、只读。未编辑正文，未训练、调参或评价，未读取 `claim_decision.json`、逐样本预测或禁止访问的正式结果。**

## 总体判断

**需重大修改后再进行 JIM 格式化。**

可核对的核心数值均与冻结表格和正式汇总一致，没有发现抄录错误、比较方向颠倒或结果选择性删除。正文也已经主动限制了因果解释、跨材料泛化、工业安全、门控物理含义和异方差普适性。

唯一阻断项是当前区间算法与标准 conformal 有效性主张不匹配。其余问题主要涉及研究设计的准确命名、组定义、数据流和模型复现细节、邻近文献，以及 JIM 的制造决策对象。

## Blocker

### B1. 当前构造不能直接继承标准 split-conformal 的有限样本保证

当前方法使用四个 inner-fold 模型为 outer-training 组生成 OOF 均值和尺度预测，再从这些折特异预测形成校准分数；测试区间的中心和尺度则来自在完整 outer-training universe 上重拟合的模型。校准组分数与测试组分数并不是同一固定预测器下的可交换分数。

因此，虽然

\[
k=\lceil(G_{\mathrm{cal}}+1)(1-\alpha)\rceil
\]

是 split conformal 常用的有限样本秩修正，但这个秩公式本身不能修复“OOF 折模型校准、全数据重拟合模型测试”的预测器不一致。当前 90.88% 和 96.86% 可以报告为嵌套外层检验中的经验同时组覆盖率，不能直接解释为标准 split conformal 的精确名义保证。

该判断由方法文献支持：

- Barber et al., *Predictive inference with the jackknife+*：原始 jackknife 只以完整拟合预测为中心、用留一残差定宽，通常不具备普适覆盖保证；jackknife+ 的关键修正是同时使用留一模型对测试点的预测。[论文](https://arxiv.org/abs/1905.02928)
- Gasparin and Ramdas, *Improving the Statistical Efficiency of Cross-Conformal Prediction*：cross-conformal 的保证形式与标准 split conformal 不同，需要专门的交叉组合构造。[论文](https://proceedings.mlr.press/v267/gasparin25a.html)

### 两种合规处理路线

#### 路线 A：保留现有计算结果，不重跑

- 将 `group-conformal prediction` 统一改为 `cross-fitted group-max interval calibration`；
- 题目、摘要、贡献 3、Figure 2、方法、讨论和结论均改为经验性区间校准；
- 明确声明标准 split-conformal 的有限样本 \(1-\alpha\) 保证不适用；
- 只报告在嵌套 outer-test 组上观察到的经验覆盖率；
- 删除或改写 `finite-sample corrected`、`confirmed validity`、`delivered the target` 等算法级保证暗示。

这条路线无需重新采集实验，也无需重跑正式结果，但会削弱题目中的 conformal 方法贡献。

#### 路线 B：纠正概率校准与评价流程

- 使用独立的组级 calibration split，或实现正确的 group-level CV+/cross-conformal 测试点聚合；
- 在不调参、不删种子/折、不覆盖旧正式根的前提下，以新的纠正分析根重新完成概率校准和评价；
- 在运行前冻结并时间戳记录纠正协议；
- 旧结果保留为审计记录，新结果明确标注为对统计有效性问题的纠正，不伪装为原预注册分析。

这条路线不需要新增物理实验，也不要求重新优化均值模型；是否需要重新拟合折特异概率模型，须在检查现有检查点与预测接口后确定。

## High-priority findings

### H1. Phase B 不是独立数据确认

Phase A 和 Phase B 使用相同的 212 个组及相同外层结构，且 Phase B 包含 Phase A 的种子。Phase B 应改称 `same-data, multi-seed robustness replication`，不能暗示独立外部复现。Bootstrap 区间也不包含 Phase A 架构选择带来的选择后不确定性。

### H2. `group_id` 与组间独立性缺少操作定义

需说明一个组对应一次切削、一个槽、一个工件还是一个工艺组合，以及不同组是否共享刀具、工件批次、加工顺序和磨损历史。在无法证明独立性时，把 `independent machining groups` 改为 `treated as the indivisible split and resampling unit`。

### H3. 数据纳入与振动窗口形成过程不足以复现

需报告 212 个组如何形成 586 个区域、每组区域数分布、无效窗口排除、有效性判据、区域与信号片段的时间对齐、每区域窗口数量。组内观测数不同会影响 group-max 分数，这一点必须披露。

### H4. 基线与对照模型细节不足

需补充 M0 的 ridge 惩罚或选择方式，以及 V1/F1/R1/G1 的网络头、参数量、训练与冻结顺序和超参数来源；删除只对内部实现有意义的 `loaded read-only`。

### H5. 邻近文献覆盖不足

Related Work 和证据矩阵至少需要补充：

- [Dynamical uncertainty-aware data fusion for surface roughness prediction](https://doi.org/10.1016/j.apm.2026.117136)；
- [Fine-grained fusion and uncertainty quantification considering chatter](https://doi.org/10.1016/j.engappai.2025.113550)；
- [Two-stage multi-source fusion for roughness prediction](https://doi.org/10.1016/j.engappai.2026.115291)；
- jackknife+/CV+/cross-conformal 有效性文献。

定位应突出“强工艺锚点下的阈值损害评估和组层级区间目标”，而不是一般动态加权或一般不确定性感知融合。

### H6. `Safe` 仍可能被误解为理论保证

Sigmoid 门控不会精确回退为零，当前算法也没有逐组误差增量约束，Phase B 的 G1 仍有 20.91% 的阈值负迁移。建议题目使用 `Selective` 或 `Process-Anchored`；Figure 1 的 `Trust gate` 改为 `Correction gate`，`safe fallback` 改为 `process-reference limiting path`。

### H7. `pre-registered` 用词需要外部可验证依据

如能提供公开 Git commit、OSF/Zenodo 或第三方时间戳，可保留 preregistration；否则统一改为 `prospectively frozen analysis protocol before formal evaluation`。

## Medium-priority findings

1. `material negative transfer` 易暗示工程显著性；建议改为 `pre-specified threshold-exceeding negative-transfer rate`，或明确 `material` 只是操作标签。
2. 覆盖率是同一 212 个组的三个种子特异覆盖率的算术平均，不是 636 个独立组；正文和 Figure 5 应写清分母与依赖。
3. 90.88% 和 96.86% 是经验点估计，没有覆盖率置信区间；应写为接近或高于名义水平的经验结果，不说“证明有效”。
4. Bootstrap 需说明区间类型、折汇总方式、三种子处理顺序，以及区间只条件于已训练模型，不包含重训练、架构选择和折划分不确定性。
5. 门控分位数和 v3/v4 MAE 尚未落到允许核对的 Table 1–5；应加入补充表或删去次要解释。
6. 摘要应说明 Phase A 中二次 ridge M0 仍是最强点预测基线，避免让读者以为 P1 是全体最佳基线。
7. JIM 制造决策对象需具体限定为离线模型审计、决定是否保留传感器增量、为完整加工运行安排复检；不是替代测量或闭环控制。
8. 排版前必须确定数据、代码、固定组划分和配置文件的实际共享方式。

## 可留到 JIM 格式化阶段

- 摘要压缩到 150–250 词，目标 230–245 词；
- 关键词由 8 个减至 4–6 个；
- 编号引文转换为作者—年份；
- 把内部来源名换成公开仓库、补充材料编号或方法位置；
- 将 Figure 3 复制为可移植投稿图件；
- 统一 Bootstrap、\(R_a\) 和单位格式；
- 将 `9 registered inputs` 改为 `3 process variables expanded to 9 quadratic features`；
- 补齐 title page、作者单位、基金、贡献、利益冲突和 ORCID。

## 是否需要重跑

- **不需要重新采集实验。**
- **不需要重新训练或调参均值模型。**
- 接受路线 A 时，无需重跑任何正式结果。
- 坚持标准 group-conformal 有限样本保证时，需要执行路线 B 的纠正概率校准和评价；这是对既有数据的重新分析，必须由作者另行授权，并且不得覆盖旧正式结果。

