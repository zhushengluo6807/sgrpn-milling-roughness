from pathlib import Path

import pytest

from roughness.sgrpn import cli


def test_parser_exposes_only_registered_phase_a_commands():
    parser = cli.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {"audit", "features", "train-phase-a", "evaluate-phase-a", "run-phase-a"}
    train = parser.parse_args(["train-phase-a", "--config", "x.yaml", "--fold", "0", "--device", "auto", "--resume"])
    assert train.fold == 0
    assert train.device == "auto"
    assert train.resume is True


def test_evaluate_never_calls_training(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events = []
    monkeypatch.setattr(cli, "load_sgrpn_config", lambda path: object())
    monkeypatch.setattr(cli, "_evaluate", lambda config: events.append("evaluate"))
    monkeypatch.setattr(cli, "_train", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrained")))

    assert cli.main(["evaluate-phase-a", "--config", str(tmp_path / "cfg.yaml")]) == 0
    assert events == ["evaluate"]


def test_run_phase_a_orders_all_steps_and_never_runs_phase_b(monkeypatch: pytest.MonkeyPatch):
    events = []
    config = object()
    monkeypatch.setattr(cli, "load_sgrpn_config", lambda path: config)
    monkeypatch.setattr(cli, "_audit", lambda value: events.append("audit"))
    monkeypatch.setattr(cli, "_features", lambda value: events.append("features"))
    monkeypatch.setattr(cli, "_train", lambda value, **kwargs: events.append("train"))
    monkeypatch.setattr(cli, "_evaluate", lambda value: events.append("evaluate"))

    assert cli.main(["run-phase-a", "--config", "cfg.yaml", "--device", "cpu", "--resume"]) == 0
    assert events == ["audit", "features", "train", "evaluate"]


def test_cli_returns_nonzero_for_incompatible_formal_artifacts(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(cli, "load_sgrpn_config", lambda path: object())
    monkeypatch.setattr(cli, "_evaluate", lambda config: (_ for _ in ()).throw(ValueError("OOF Cartesian incomplete")))

    assert cli.main(["evaluate-phase-a", "--config", "cfg.yaml"]) != 0
    assert "OOF Cartesian incomplete" in capsys.readouterr().err
