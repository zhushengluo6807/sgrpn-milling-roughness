from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from roughness.scheme1_physics.config import (
    EXPECTED_MODELS,
    load_scheme1_physics_config,
    write_physics_protocol_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = PROJECT_ROOT / "configs" / "scheme1.yaml"


def _write_config(
    tmp_path: Path,
    *,
    output_dir: Path,
    source_config: Path = SOURCE_CONFIG,
) -> Path:
    path = tmp_path / "scheme1_physics.yaml"
    path.write_text(
        "\n".join(
            [
                f"scheme1_config_path: {source_config.as_posix()}",
                f"output_dir: {output_dir.as_posix()}",
                "models: [PW0, PW1, PW2, PE0, PE1, PE2]",
                "batch_size: 4",
                "validation_fraction: 0.2",
                "residual_dropout: 0.3",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_load_config_reuses_source_and_keeps_output_isolated(tmp_path: Path):
    output_dir = tmp_path / "physics"
    config = load_scheme1_physics_config(
        _write_config(tmp_path, output_dir=output_dir)
    )
    assert config.output_dir == output_dir.resolve()
    assert config.output_dir != config.source.output_dir
    assert config.models == EXPECTED_MODELS
    assert config.batch_size == 4


def test_config_rejects_source_output_as_destination(tmp_path: Path):
    from roughness.scheme1.config import load_scheme1_config

    source = load_scheme1_config(SOURCE_CONFIG)
    path = _write_config(tmp_path, output_dir=source.output_dir)
    with pytest.raises(ValueError, match="must differ"):
        load_scheme1_physics_config(path)


def test_protocol_checkpoint_records_isolation(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "roughness.scheme1_physics.config.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    config = load_scheme1_physics_config(
        _write_config(tmp_path, output_dir=tmp_path / "physics")
    )
    output = write_physics_protocol_checkpoint(config)
    assert output == config.output_dir / "run_manifest.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["git_available"] is True
    assert "PW2" in payload["models"]
    assert payload["source_output_dir"] == str(config.source.output_dir)
