import pytest

from roughness.scheme1_physics.cli import (
    build_parser,
    ordered_neural_models,
)


def test_train_cli_defaults_to_four_neural_models():
    args = build_parser().parse_args(
        ["train", "--config", "configs/scheme1_physics.yaml"]
    )
    assert args.models == "PW1,PW2,PE1,PE2"
    assert args.folds == "0,1,2,3,4"
    assert args.seeds == "20260723,20260724,20260725"
    assert args.batch_size == 4


def test_run_orders_ordinary_before_gated_and_adds_parent():
    assert ordered_neural_models(
        ["PW2", "PE2", "PE1", "PW1"]
    ) == ["PW1", "PE1", "PW2", "PE2"]
    assert ordered_neural_models(["PW2"]) == ["PW1", "PW2"]
    assert ordered_neural_models(["PE2"]) == ["PE1", "PE2"]


def test_model_order_rejects_baseline_and_unknown_names():
    with pytest.raises(ValueError, match="neural"):
        ordered_neural_models(["PW0"])
