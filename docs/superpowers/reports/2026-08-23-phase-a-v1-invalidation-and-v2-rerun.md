# Phase A v1 invalidation and v2 formal rerun

## Status

The only authoritative Phase A decision is the fresh `sgrpn-phase-a-v2` result emitted
on 2026-08-23. It permits Phase B through the preregistered `transfer_safety` path.

The earlier v1 decision is invalid and must not be cited as evidence or used to authorize
Phase B. Its artifacts are retained only for audit at:

`outputs/sgrpn/phase_a_invalid_v1_swap_selection_20260823`

## Why v1 was invalidated

The v1 formal inference predictor averaged the original and Ch9/Ch10-swapped orders for
V1, F1, R1, and G1. Early stopping, however, selected checkpoints from validation loss
computed only in the original channel order. Model selection therefore optimized a
different predictor from the one evaluated formally. This was a protocol-level defect,
so all v1 checkpoints, OOF predictions, evaluation results, and the v1 acceptance
decision were declared non-authoritative.

Commit `43077a3` aligned validation-time early stopping with the swap-averaged inference
predictor. Commit `e99140f` introduced the centralized `sgrpn-phase-a-v2` protocol and
bound it into fingerprints, states, completion markers, checkpoints, histories, and
scaler metadata. Missing or v1 protocol metadata is rejected by resume and deep
validation. Commit `0c1efa6` also aligned the Phase B correction feature specification
with the swap-averaged `ModelOutput`.

## Earlier evaluation abort and technical retry

The original v1 evaluation first aborted before metrics because an exact pandas frame
comparison detected a one-ULP CSV round-trip difference in `absolute_difference_s`
(maximum `8.326672684688674e-17`). No acceptance file was written by that failed run.
Commit `693787e` made duration-audit persistence round-trip stable. One technical retry
then completed, but that v1 result was later superseded in full by the independent
swap-selection defect above. The parser retry was not a tuning iteration.

## Fresh v2 execution evidence

- Source HEAD for the run: `0c1efa6d0ad8e1291b86ce4e6c1613b1e5a84499`.
- Fresh fingerprint: `a72c236bea4e51a1a921f4f7ad944a326aedf3957dd768ab6ad547097ea21339`.
- Fresh audit/features: 212 groups, 586 segments, 5 folds, 3 channels, 25,600 Hz,
  361 bins, 1,673 windows, and 86 registered duration mismatches.
- Five clean CUDA folds completed P1/V1/F1/R1/G1 under `sgrpn-phase-a-v2` in a
  single training command (exit 0; 4,415.858 seconds).
- Exactly one v2 evaluation command ran (exit 0; 25.896 seconds); it required no
  technical retry.
- Deep validation proved 5 markers, 5 states, and 75 registered stage artifacts have
  valid hashes and matching v2 protocol/fingerprint bindings.
- OOF validation proved 3,516 rows (`586 x 6`), no duplicate sample/model pairs, finite
  registered metrics, valid gates, and five group-level Bootstrap comparisons with
  10,000 repetitions each.
- The SGRPN suite passed: `245 passed in 37.23s`.
- All 640 legacy files were byte-identical before and after. Their registered hash-map
  SHA-256 was
  `a60feee07e10bbc5c1d713a02b03a512c5d2d6cc284af159a04286c16b298a1e`.

## Authoritative v2 decision

The registered acceptance file was read exactly once after the post-evaluation checks:

- `proceed_to_phase_b`: `true`
- `path`: `transfer_safety`
- `process_expert_credible`: `true`
- `g1_noninferior`: `true`
- `mean_improvement_path`: `false`
- `transfer_safety_path`: `true`
- `gate_collapsed_without_benefit`: `false`

The decision was reported as emitted. No v1-v2 metric comparison, outcome-driven model
selection, hyperparameter tuning, or Phase B execution was performed during the rerun.

