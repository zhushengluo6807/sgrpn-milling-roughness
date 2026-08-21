from pathlib import Path

import pytest

from roughness.sgrpn.config import load_sgrpn_config


def test_phase_a_protocol_is_locked():
    cfg = load_sgrpn_config(Path("configs/sgrpn_phase_a.yaml"))
    assert cfg.sample_rate_hz == 25600
    assert cfg.window_samples == 25600
    assert cfg.order_min == 0.0
    assert cfg.order_max == 90.0
    assert cfg.order_step == 0.25
    assert cfg.seeds == (20260723,)
    assert cfg.inner_splits == 4
    assert cfg.bootstrap_repetitions == 10000


def test_rejects_phase_a_with_multiple_seeds(tmp_path):
    text = Path("configs/sgrpn_phase_a.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(text.replace("seeds: [20260723]", "seeds: [1, 2]"), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly seed 20260723"):
        load_sgrpn_config(path)
