# Final branch review fix report

Base: `b5a02fb`
Fix commit: `3c33121` (`fix: validate phase B calibration derivation`)

## Scope

Functional changes are limited to the Phase B deep calibration validator and its
existing training test module. No Phase A/legacy code, deferred-minor refactoring,
reporting integration, or formal Phase B run was performed.

## Root cause and fix

Generation derives group scores through `_calibration_group_scores` and derives
finite-sample quantiles from those scores. The completed-artifact validator had
validated persisted nested OOF, group scores, and JSON quantiles independently,
which allowed coordinated content-and-hash tampering to appear complete.

The validator now recomputes canonical score tables per registered scale model from
persisted OOF, requires the canonical row/metadata order and exact provenance/count
fields, and compares scores with only `1e-12` absolute, zero-relative CSV
round-trip tolerance. It then recomputes each registered finite-sample quantile
from those canonical scores and verifies JSON alpha, group count, one-based order
index, and quantile. The same narrow tolerance is used for the selected quantile,
because OOF CSV parsing can change an order statistic by one final binary digit.

## TDD evidence

- RED: `E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_phase_b_training.py -k semantic_mismatch_after_hash_refresh -q`
  produced `2 failed, 44 deselected`. Both tests refreshed the affected completion
  artifact hash; the old deep validator incorrectly returned a completed match.
- GREEN: the same focused command produced `2 passed, 44 deselected` after the
  minimal validator change.

The regressions exercise the public completed-artifact boundary:

1. A nested OOF `mu` is forged while its completion hash is refreshed; the stale
   persisted score table is now rejected as non-derived.
2. A persisted JSON order index and quantile are forged while its completion hash
   is refreshed; the score-derived finite-sample result is now rejected.

## Verification

- `E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn -q`:
  `433 passed, 2 skipped in 214.19s`.
- `D:\CodexPython\python.exe -m compileall -q src tests/sgrpn`: exit 0.
- `git diff --check`: exit 0 before staging; `git diff --cached --check`: exit 0
  before the fix commit.
- `E:\CodeX\机床项目\.venv\Scripts\roughness-sgrpn.exe preflight-phase-b --config configs/sgrpn_phase_b.yaml`
  completed read-only and emitted the validated Phase A handoff hashes.
- `rg --files outputs/sgrpn/phase_b` confirmed the directory contains only
  `outputs/sgrpn/phase_b/test_evidence.json`.

## Concerns

None for the code change. The configured default Python lacked both `pytest` and
`torch`; repository tests therefore used the existing workspace virtual environment.
