import json
import hashlib
from types import SimpleNamespace

import pytest


def test_train_runs_registered_cartesian_then_assembles_once(monkeypatch, capsys):
    from roughness.sgrpn import corrective_cli

    corrective_config = SimpleNamespace()
    phase_b_config = SimpleNamespace(seeds=(20260723, 20260724, 20260725))
    context = (corrective_config, phase_b_config, object(), object(), object())
    trained: list[tuple[int, int, str]] = []
    assembled: list[str] = []

    monkeypatch.setattr(corrective_cli, "_load_context", lambda path: context)
    monkeypatch.setattr(corrective_cli, "_selected_device", lambda requested: "cuda")
    monkeypatch.setattr(
        corrective_cli,
        "run_corrective_fold",
        lambda *args, fold, seed, device: trained.append((fold, seed, device)),
    )
    monkeypatch.setattr(
        corrective_cli,
        "run_corrective_all",
        lambda *args, device: assembled.append(device),
    )

    assert corrective_cli.main(["train", "--config", "corrective.yaml", "--device", "auto"]) == 0
    assert trained == [
        (fold, seed, "cuda")
        for fold in range(5)
        for seed in (20260723, 20260724, 20260725)
    ]
    assert assembled == ["cuda"]
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [message["completed_units"] for message in messages[:-1]] == list(range(1, 16))
    assert messages[-1] == {
        "completed_units": 15,
        "device": "cuda",
        "training_status": "complete",
    }


@pytest.mark.parametrize(
    "argv",
    (
        ["train", "--config", "corrective.yaml", "--fold", "0"],
        ["train", "--config", "corrective.yaml", "--seed", "20260723"],
    ),
)
def test_train_requires_fold_and_seed_together(argv):
    from roughness.sgrpn import corrective_cli

    with pytest.raises(SystemExit):
        corrective_cli.main(argv)


def test_evaluate_calls_one_shot_entry_without_printing_claim(monkeypatch, capsys):
    from roughness.sgrpn import corrective_cli

    context = (object(), object(), object(), object(), object())
    calls: list[str] = []
    monkeypatch.setattr(corrective_cli, "_load_context", lambda path: context)
    monkeypatch.setattr(
        corrective_cli,
        "evaluate_corrective_once",
        lambda corrective, phase_b, bundle, cache: calls.append("called"),
        raising=False,
    )

    assert (
        corrective_cli.main(
            [
                "evaluate",
                "--config",
                "corrective.yaml",
            ]
        )
        == 0
    )
    assert calls == ["called"]
    assert json.loads(capsys.readouterr().out) == {
        "evaluation_invocations": 1,
        "evaluation_status": "complete",
    }


def test_load_context_rejects_phase_a_config_hash_drift(tmp_path, monkeypatch):
    from roughness.sgrpn import corrective_cli

    phase_a_path = tmp_path / "phase-a.yaml"
    phase_a_path.write_text("max_epochs: 999\n", encoding="utf-8")
    actual = hashlib.sha256(phase_a_path.read_bytes()).hexdigest()
    assert actual != "0" * 64
    monkeypatch.setattr(
        corrective_cli,
        "load_corrective_config",
        lambda path: SimpleNamespace(phase_b_config_path=tmp_path / "phase-b.yaml"),
    )
    monkeypatch.setattr(
        corrective_cli,
        "load_phase_b_config",
        lambda path: SimpleNamespace(
            phase_a_config_path=phase_a_path,
            phase_a_config_file_sha256="0" * 64,
        ),
    )

    with pytest.raises(ValueError, match="Phase A config hash"):
        corrective_cli._load_context("corrective.yaml")
