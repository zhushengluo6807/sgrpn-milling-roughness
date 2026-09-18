# SGRPN manuscript tables

## Table 1. Dataset and experimental structure.

| item | value | definition_or_scope | source |
| --- | --- | --- | --- |
| Independent machining groups | 212 | Outer splitting and Bootstrap resampling unit | Phase A run manifest |
| Surface regions | 586 | Prediction unit; regions remain nested in machining groups | Phase A run manifest |
| Ra readings per region | 3 | Repeated readings share one region-level mean and uncertainty model | Locked SGRPN protocol |
| Vibration channels | 3 (Ch9, Ch10, Ch11) | Ch11 is axial; Ch9/Ch10 orientation is unresolved and symmetrized | Locked SGRPN protocol |
| Vibration sampling rate | 25.6 kHz | Each retained model window contains 25,600 samples (1 s) | Locked SGRPN protocol |
| Spindle-speed grid | 10 levels (4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500 rpm) | Observed levels in the frozen Phase A breakdown | error_by_n_rpm.csv |
| Feed-per-tooth grid | 5 levels (0.03, 0.06, 0.09, 0.12, 0.15 mm/tooth) | Observed levels in the frozen Phase A breakdown | error_by_fz_mm_per_tooth.csv |
| Axial-depth grid | 4 levels (0.5, 1.0, 1.5, 2.0 mm) | Observed levels in the frozen Phase A breakdown | error_by_ap_mm.csv |
| Observed process combinations | 189 / 200 | Observed combinations relative to the full 10×5×4 grid | Locked SGRPN data specification |
| Acquisition versions | 2 (v3, v4) | Composite-domain stress test; version is excluded from model inputs | error_by_version.csv |
| Outer cross-validation | 5 group-disjoint folds | Recorded group overlap count: 0 | Phase A run manifest |
| Registered random seeds | 1 in Phase A; 3 in Phase B | Phase A feasibility followed by frozen three-seed replication | Phase A/Phase B protocols |

## Table 2. Registered model and comparator matrix.

| model | architecture | inputs | training_target | role |
| --- | --- | --- | --- | --- |
| M0 | Quadratic process ridge regression | Process parameters and registered quadratic terms | Region-mean Ra | Strong classical process baseline |
| P1 | Process MLP | Nine scaled process features | Region-mean Ra | Neural process expert and safe fallback |
| V1 | Vibration-only CNN | Three-channel order spectra | Region-mean Ra | Tests vibration without process context |
| F1 | Direct process-vibration fusion | Process features and order-spectrum embedding | Region-mean Ra | Naive fusion comparator |
| R1 | Ungated residual fusion | P1 process mean and order-spectrum embedding | Group-safe P1 OOF residual | Tests the full learned vibration correction |
| G1 | Selective gated residual fusion | P1 mean, residual embedding, process and seven quality features | Group-safe P1 OOF residual with gated correction | Proposed safe fusion model |

## Table 3. Point prediction and material negative transfer.

| phase | seed_scope | model | reference | mae_um | rmse_um | r2 | fold_wins_vs_reference | material_negative_transfer_rate | mae_improvement_um | mae_improvement_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase A | 20260723 | M0 | — | 0.104619 | 0.142870 | 0.826172 | — | — | — | — |
| Phase A | 20260723 | P1 | M0 | 0.107735 | 0.147107 | 0.815708 | 1/5 | — | -0.003116 | [-0.008250, 0.001994] |
| Phase A | 20260723 | V1 | P1 | 0.175927 | 0.233567 | 0.535420 | 0/5 | 69.34% | -0.068192 | [-0.086720, -0.050499] |
| Phase A | 20260723 | F1 | P1 | 0.119040 | 0.167782 | 0.760267 | 1/5 | 47.64% | -0.011305 | [-0.022760, -0.000647] |
| Phase A | 20260723 | R1 | P1 | 0.110505 | 0.147564 | 0.814561 | 2/5 | 35.85% | -0.002770 | [-0.009456, 0.003620] |
| Phase A | 20260723 | G1 | P1 | 0.106314 | 0.145085 | 0.820740 | 2/5 | 19.81% | 0.001421 | [-0.002421, 0.005166] |
| Phase B | mean of 3 registered seeds | P1 | — | 0.108075 | 0.147046 | 0.815846 | — | — | — | — |
| Phase B | mean of 3 registered seeds | R1 | P1 | 0.112636 | 0.150994 | 0.805644 | — | 37.11% | -0.004561 | [-0.010636, 0.001247] |
| Phase B | mean of 3 registered seeds | G1 | P1 | 0.108149 | 0.145847 | 0.818749 | — | 20.91% | -0.000074 | [-0.003133, 0.002817] |

Note: Positive MAE improvement means that the candidate has lower MAE than its reference. Material negative transfer uses a 0.01 µm group-MAE excess threshold. Phase B fold-win counts are intentionally omitted because the registered replication summary treats seeds as replicates.

## Table 4. All-seed group-conformal results.

| scale_model | nominal_coverage | single_reading_coverage | simultaneous_group_coverage | mean_interval_width_um | winkler_score | gaussian_nll | gaussian_crps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Homoscedastic | 90.00% | 94.86% | 90.88% | 0.577811 | 0.703717 | -0.419528 | 0.080510 |
| Homoscedastic | 95.00% | 97.60% | 96.86% | 0.784051 | 0.917948 | -0.419528 | 0.080510 |
| Heteroscedastic | 90.00% | 95.36% | 92.45% | 0.818010 | 0.913044 | 15.607432 | 0.083007 |
| Heteroscedastic | 95.00% | 97.76% | 96.23% | 1.400525 | 1.506541 | 15.607432 | 0.083007 |

Note: Coverage applies to exchangeable new machining groups of the existing type. NLL and CRPS do not depend on the nominal conformal level and are repeated to keep each row self-contained.

## Table 5. Paired group-Bootstrap ablation of heteroscedastic versus homoscedastic scale.

| metric | nominal_coverage | heteroscedastic_minus_homoscedastic | ci95_lower | ci95_upper | ci_excludes_zero | preferred_direction | observed_direction | registered_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian NLL | — | 16.026960 | 0.353873 | 46.285937 | Yes | Lower is better | Heteroscedastic is worse | No heteroscedastic advantage |
| Gaussian CRPS | — | 0.002496 | 0.001031 | 0.003999 | Yes | Lower is better | Heteroscedastic is worse | No heteroscedastic advantage |
| Mean interval width | 90.00% | 0.240198 | 0.195791 | 0.286235 | Yes | Lower is better | Heteroscedastic is worse | No heteroscedastic advantage |
| Mean interval width | 95.00% | 0.616475 | 0.531795 | 0.706250 | Yes | Lower is better | Heteroscedastic is worse | No heteroscedastic advantage |
| Winkler score | 90.00% | 0.209327 | 0.133687 | 0.280637 | Yes | Lower is better | Heteroscedastic is worse | No heteroscedastic advantage |
| Winkler score | 95.00% | 0.588593 | 0.444887 | 0.724953 | Yes | Lower is better | Heteroscedastic is worse | No heteroscedastic advantage |

Note: Differences are heteroscedastic minus homoscedastic; all listed metrics are lower-is-better. A positive interval excluding zero therefore disfavors the heteroscedastic scale model.
