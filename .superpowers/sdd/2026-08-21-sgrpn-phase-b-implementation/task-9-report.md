# Task 9 report — reproducible Phase B outputs and CLI commands

## Status

DONE

## RED evidence

Command:

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_reporting.py tests/sgrpn/test_cli.py -k phase_b -v
```

Raw result summary: collection failed as intended with `ImportError: cannot import name 'write_phase_b_report' from roughness.sgrpn.reporting`; 34 tests were deselected and no Phase B reporting API existed.

## GREEN and verification evidence

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn/test_reporting.py tests/sgrpn/test_cli.py -v
```

Result: `48 passed in 24.09s`.

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m pytest tests/sgrpn -q
```

Result: `425 passed, 2 skipped in 156.04s`.

```powershell
E:\CodeX\机床项目\.venv\Scripts\python.exe -m compileall -q src/roughness/sgrpn tests/sgrpn
git diff --check
Test-Path outputs/sgrpn/phase_b
```

Result: compile and diff checks exited `0`; the Phase B output root check returned `False`. No formal Phase B training or evaluation was run.

## Changes

- Added Phase B report publication, deterministic CSV/JSON reconstruction, persisted-table-only figures, input hashing, and deep output validation in `src/roughness/sgrpn/reporting.py`.
- Added preflight, train, evaluate, and resumable run CLI commands with handoff-first gating in `src/roughness/sgrpn/cli.py`.
- Added synthetic OOF reconstruction/immutability coverage and CLI gate/order/resume coverage in the two allowed test modules.

## Commit

`99618ca feat: report SGRPN phase B results`

## Self-review and residual concerns

- The report validator reloads all fold checkpoints and calibration artifacts, recomputes the OOF Cartesian/quantile/interval/metric/bootstrap/fingerprint checks, and requires all 15 fold-seed completions before accepting a formal output.
- Selected devices are captured in the Phase B run manifest at training time because the prior fold-state schema intentionally does not persist a device field.
- No formal artifact was created; formal Phase B execution remains a separate, explicitly authorized task.
