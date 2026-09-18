# SGRPN manuscript asset notes

## Technical summary

The paper-ready assets preserve the registered evidence boundary: the strongest mean-model result is lower material negative transfer, not a statistically clear point-accuracy gain. The retained uncertainty result is simultaneous group-conformal coverage for exchangeable new machining groups. The heteroscedastic scale ablation is reported as a negative result.

## Figure 1 contract and caption

- Analytical question: How does SGRPN combine a safe process-only baseline with a selectively gated vibration correction?
- Form: Left-to-right module diagram with explicit signal paths and mathematical operators.
- Evidence boundary: This is a method schematic, not an empirical result or a causal explanation of the learned gate.
- Safety path: When the gate closes, the model falls back toward the P1 process expert.
- Non-color encoding: Labels, arrows, multiplication/addition nodes, and dashed control links preserve meaning without color.

Suggested caption: **Architecture of the selectively gated residual process network (SGRPN).** The P1 process expert supplies the safe fallback mean prediction. A residual CNN derives a vibration-conditioned correction, while the trust gate controls how much of that correction is admitted. The final predictor is the P1 output plus the gated residual. The diagram specifies the computational structure only; it does not imply that the learned gate is a causal measure of vibration quality.

## Figure 2 contract and caption

- Analytical question: Where are training, calibration, and outer testing separated to prevent group leakage?
- Form: Hierarchical workflow with outer train/test partitioning and an inner calibration split.
- Unit of isolation: Machining group; readings from one group never cross a split boundary.
- Calibration rule: A group score is the maximum standardized residual within that calibration group.
- Evaluation rule: The outer-test groups are evaluated once after model fitting and conformal calibration.

Suggested caption: **Group-isolated nested training and conformal calibration workflow.** Each outer fold holds out complete machining groups. The remaining groups are separated into model-fitting and conformal-calibration subsets, with no reading-level crossover. Calibration reduces each group to its maximum standardized residual before computing the registered conformal quantile. The calibrated predictor is then applied once to the untouched outer-test groups, and results are aggregated across registered folds and seeds.

## Figure 4 contract and caption

- Analytical question: Does selective gating reduce material negative transfer relative to direct and ungated vibration fusion?
- Form: Two-panel categorical bar chart with a shared zero-based rate axis.
- Denominator: 212 machining groups; Phase B is summarized across three registered seeds.
- Definition: A group is materially harmed when its weighted MAE exceeds P1 by more than 0.01 µm.
- Non-color encoding: Direct labels and hatching distinguish comparator bars from G1.

Suggested caption: **Material negative-transfer rate relative to the P1 process expert.** A group is counted as materially negatively transferred when its weighted group-level MAE exceeds P1 by more than 0.01 µm. Phase A uses the registered feasibility seed; Phase B reports the mean across three registered seeds. Lower values indicate safer fusion.

Interpretation paragraph: Selective gating reduced the Phase A material negative-transfer rate to 19.81%, compared with 47.64% for direct fusion and 35.85% for ungated residual fusion. The three-seed Phase B summary remained directionally consistent (20.91% for G1 versus 37.11% for R1). These results support a fusion-safety claim, not a causal interpretation of the gate or a claim of superior average point accuracy.

## Figure 5 contract and caption

- Analytical question: Does group conformal calibration meet the intended simultaneous coverage, and what interval-width cost accompanies each scale model?
- Form: Coverage line-and-point panel plus a separate zero-based width panel; no dual axis.
- Population: All-seed summaries over 212 machining groups.
- Benchmark: The neutral dotted line marks nominal 90% and 95% coverage.
- Scale disclosure: The coverage panel uses a focused 88–100% y-axis; the width panel starts at zero.
- Non-color encoding: Marker shape, line style, open fill, and hatching complement color.

Suggested caption: **Group-conformal coverage and interval width.** Empirical simultaneous group coverage is shown against the nominal 90% and 95% levels, with the corresponding mean interval widths in a separate panel. Results apply only to exchangeable new machining groups of the existing type. The homoscedastic model attained 90.88% and 96.86% group coverage with mean widths of 0.578 and 0.784 µm, respectively; heteroscedastic intervals were wider without a registered scoring advantage.

Interpretation paragraph: Both calibrated models reached approximately nominal simultaneous group coverage, but the heteroscedastic scale required substantially wider intervals. Combined with worse Winkler, Gaussian NLL, and CRPS values, this supports retaining group conformal calibration while declining to emphasize heteroscedasticity.

## Figure 6 contract and caption

- Analytical question: Does the heteroscedastic scale improve any registered lower-is-better uncertainty metric over the homoscedastic scale?
- Form: Four-panel paired group-Bootstrap forest plot because the metric scales differ materially.
- Contrast: Every point is heteroscedastic minus homoscedastic; positive values disfavor heteroscedasticity.
- Uncertainty: Horizontal bars are registered 95% paired group-Bootstrap intervals from 10,000 resamples.
- Reference: The vertical zero line denotes no difference.

Suggested caption: **Registered paired group-Bootstrap ablation of the heteroscedastic scale model.** Points show heteroscedastic-minus-homoscedastic differences and horizontal bars show 95% paired group-Bootstrap intervals from 10,000 group resamples. All metrics are lower-is-better, so positive differences disfavor the heteroscedastic scale. The intervals for Gaussian NLL, CRPS, interval width, and Winkler score all lie above zero; the prespecified analysis therefore provides no support for a heteroscedastic advantage.

## Source inventory

- Phase A structure and audit: `outputs/sgrpn/phase_a/run_manifest.json`
- Phase A point metrics and folds: `outputs/sgrpn/phase_a/evaluation/summary_metrics.csv`, `fold_metrics.csv`
- Phase A negative transfer and Bootstrap: `negative_transfer.csv`, `paired_bootstrap.csv`
- Phase B mean and negative-transfer results: `outputs/sgrpn/phase_b/evaluation/mean_metrics.csv`, `negative_transfer.csv`
- Phase B uncertainty and ablation: `outputs/sgrpn/phase_b/evaluation/probability_metrics.csv`, `paired_bootstrap.csv`

The generator does not read the formal one-time claim-decision file and does not invoke training or evaluation commands.
