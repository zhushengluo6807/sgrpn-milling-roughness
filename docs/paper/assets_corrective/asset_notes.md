# SGRPN manuscript asset notes

## Technical summary

These paper-ready assets use the **post-audit corrective reanalysis**. The supported mean-model result is lower material negative transfer, not a statistically clear point-accuracy gain. Both scale variants are retained. The uncertainty claim is limited to empirical simultaneous group coverage for exchangeable new machining groups of the existing type; heteroscedasticity is not emphasized.

## Figure 1 contract and caption

**Architecture of SGRPN.** The P1 process expert supplies the fallback mean prediction. A residual CNN proposes a vibration-conditioned correction and a bounded gate controls the admitted correction. The diagram specifies computation only and does not give the gate a causal or physical interpretation.

## Figure 2 contract and caption

**Group-isolated model fitting, calibration, and testing.** Complete machining groups are assigned to proper training, fixed calibration, or outer testing. Calibration contributes one maximum standardized residual per group, and the locked predictor is applied once to outer-test groups.

## Figure 4 contract and caption

**Material negative-transfer rate relative to P1.** A group is materially harmed when candidate group MAE exceeds P1 by more than 0.01 µm. Phase A rates are 47.64% for F1, 35.85% for R1, and 19.81% for G1. The Corrective Phase B rates are 39.78% for R1 and 16.51% for G1. Lower values indicate safer fusion; the result is not a claim of superior average accuracy.

## Figure 5 contract and caption

**Group split-conformal coverage and interval width.** At nominal 90% and 95%, the homoscedastic path attained group coverage of 92.45% and 97.01%, with widths of 0.643 and 0.902 µm. The heteroscedastic path attained 93.87% and 96.70%, with widths of 1.084 and 1.919 µm. Coverage applies only to exchangeable new groups of the observed type.

## Figure 6 contract and caption

**Paired group-Bootstrap scale ablation.** Points are heteroscedastic-minus-homoscedastic differences and bars are 95% paired group-Bootstrap intervals from 10,000 group resamples. All displayed metrics are lower-is-better; positive values therefore disfavor heteroscedasticity. The frozen corrective rule retains both variants but does not permit a heteroscedastic emphasis.

## Table 6 diagnostic status

`table_6_scale_diagnostics.csv` is a post-audit descriptive analysis of the already frozen outer-held-out probability predictions. It was not part of the registered decision rule, did not trigger model fitting or subgroup selection, and must be labelled exploratory. It may be used to describe weak scale–error association and the near-zero heteroscedastic scale tail, but not to claim a uniquely identified causal failure mechanism.

## Source inventory

- Phase A: `outputs/sgrpn/phase_a/evaluation`
- Phase B source for this asset set: `outputs/sgrpn/phase_b_split_conformal_corrective/evaluation`
- Analysis status: `post-audit corrective reanalysis`

The generator does not read the formal one-time claim-decision file and does not invoke training or evaluation commands.
