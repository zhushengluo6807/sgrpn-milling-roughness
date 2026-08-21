import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from roughness.scheme1.training import (
    EarlyStopper,
    build_scheme1_model,
    make_group_train_validation_split,
    scheme1_forward,
    set_global_seed,
    train_one_fold,
    train_model,
    weighted_mae_loss,
)
from roughness.scheme1.config import Scheme1Config
from roughness.scheme1.signals import fit_channel_stats
from roughness.scheme1.models import CNNEncoder, CNNTCNEncoder


class TinyDataset(Dataset):
    def __init__(self):
        self.x = torch.arange(8, dtype=torch.float32).unsqueeze(1)
        self.y = 1.0 + 2.0 * self.x.squeeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {
            "x": self.x[index],
            "target": self.y[index],
            "sample_weight": torch.tensor(1.0),
        }


def test_weighted_mae_normalizes_by_total_weight():
    predicted = torch.tensor([0.0, 4.0])
    target = torch.tensor([1.0, 2.0])
    weight = torch.tensor([1.0, 3.0])

    loss = weighted_mae_loss(predicted, target, weight)

    assert loss.item() == 1.75


def test_group_train_validation_split_is_reproducible_and_disjoint():
    frame = pd.DataFrame(
        {
            "group_id": np.repeat([f"g{i}" for i in range(10)], 2),
            "value": np.arange(20),
        }
    )

    first = make_group_train_validation_split(
        frame, validation_fraction=0.2, seed=20260723
    )
    second = make_group_train_validation_split(
        frame, validation_fraction=0.2, seed=20260723
    )

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    train_groups = set(frame.iloc[first[0]]["group_id"])
    validation_groups = set(frame.iloc[first[1]]["group_id"])
    assert train_groups.isdisjoint(validation_groups)
    assert validation_groups


def test_early_stopper_restores_the_best_model_state():
    model = nn.Linear(1, 1, bias=False)
    stopper = EarlyStopper(patience=2)
    with torch.no_grad():
        model.weight.fill_(1.0)
    assert stopper.update(0.5, model, epoch=0)
    with torch.no_grad():
        model.weight.fill_(2.0)
    assert not stopper.update(0.6, model, epoch=1)
    with torch.no_grad():
        model.weight.fill_(3.0)
    assert not stopper.update(0.7, model, epoch=2)
    assert stopper.should_stop

    stopper.restore(model)

    torch.testing.assert_close(model.weight, torch.tensor([[1.0]]))
    assert stopper.best_epoch == 0


def test_train_model_writes_best_checkpoint_and_epoch_log(tmp_path):
    set_global_seed(20260723)
    loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
    model = nn.Linear(1, 1)

    result = train_model(
        model,
        train_loader=loader,
        validation_loader=loader,
        forward_fn=lambda current, batch: current(batch["x"]).squeeze(1),
        checkpoint_path=tmp_path / "best.pt",
        log_path=tmp_path / "history.csv",
        max_epochs=20,
        patience=5,
        learning_rate=0.05,
        weight_decay=0.0,
        device=torch.device("cpu"),
    )

    assert result.checkpoint_path.is_file()
    assert result.log_path.is_file()
    history = pd.read_csv(result.log_path)
    assert set(history.columns) == {
        "epoch",
        "train_loss",
        "validation_mae",
        "learning_rate",
        "is_best",
    }
    assert result.best_epoch == int(history.loc[history["is_best"], "epoch"].iloc[-1])
    saved = torch.load(result.checkpoint_path, weights_only=True)
    assert saved["best_epoch"] == result.best_epoch
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_scheme1_model_factory_and_forward_cover_n1_to_n4():
    batch = {
        "signal": torch.randn(2, 1, 3, 2048),
        "window_mask": torch.ones(2, 1, dtype=torch.bool),
        "process": torch.randn(2, 3),
        "physics": torch.empty(2, 0),
        "base_ra": torch.tensor([0.8, 1.0]),
    }

    for model_name in ("N1", "N2", "N3", "N4"):
        model = build_scheme1_model(model_name)
        predicted = scheme1_forward(model, batch)
        assert predicted.shape == (2,)
        assert torch.isfinite(predicted).all()


def test_fusion_model_factory_can_follow_cnn_fallback_after_tcn_gate():
    cnn = build_scheme1_model("N3", fusion_encoder="cnn")
    tcn = build_scheme1_model("N3", fusion_encoder="cnn_tcn")

    assert isinstance(cnn.encoder, CNNEncoder)
    assert isinstance(tcn.encoder, CNNTCNEncoder)


def test_train_one_fold_writes_outer_test_predictions(tmp_path):
    segments = tmp_path / "segments"
    segments.mkdir()
    manifest_rows = []
    fold_rows = []
    window_rows = []
    for index in range(6):
        path = segments / f"s{index}.csv"
        time = np.arange(2048) / 1000.0
        pd.DataFrame(
            {
                "Time_s": time,
                "Ch9_g": np.sin(2 * np.pi * (40 + index) * time),
                "Ch10_g": np.sin(2 * np.pi * (60 + index) * time),
                "Ch11_g": np.sin(2 * np.pi * (80 + index) * time),
            }
        ).to_csv(path, index=False)
        manifest_rows.append(
            {
                "sample_id": f"s{index}",
                "group_id": f"g{index}",
                "signal_path": str(path),
                "n_rpm": 4000.0 + index * 100,
                "fz_mm_per_tooth": 0.03 + index * 0.005,
                "ap_mm": 0.5,
                "ra_mean": 0.8 + index * 0.02,
                "sample_weight": 1.0,
            }
        )
        fold_rows.append(
            {
                "sample_id": f"s{index}",
                "group_id": f"g{index}",
                "fold": index % 3,
            }
        )
        window_rows.append(
            {
                "segment_id": f"s{index}",
                "group_id": f"g{index}",
                "csv_path": str(path),
                "window_id": 0,
                "start_sample": 0,
                "end_sample": 2048,
                "is_tail_aligned": False,
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    folds = pd.DataFrame(fold_rows)
    output = tmp_path / "output"
    (output / "folds").mkdir(parents=True)
    manifest_path = tmp_path / "manifest.csv"
    folds_path = tmp_path / "folds.csv"
    manifest.to_csv(manifest_path, index=False)
    folds.to_csv(folds_path, index=False)
    pd.DataFrame(window_rows).to_csv(output / "window_index.csv", index=False)
    train_ids = set(folds.loc[folds["fold"] != 0, "sample_id"])
    stats = fit_channel_stats(manifest, train_ids)
    (output / "folds" / "fold_0_channel_stats.json").write_text(
        json.dumps(stats.to_dict()),
        encoding="utf-8",
    )
    config = Scheme1Config(
        segments_dir=segments,
        manifest_path=manifest_path,
        folds_path=folds_path,
        output_dir=output,
        sample_rate_hz=1000,
        window_samples=2048,
        stride_samples=2048,
        re_candidates_mm=(0.2,),
        seeds=(20260723,),
        inner_splits=2,
        welch_nperseg=512,
        welch_noverlap=256,
        nominal_band_min_halfwidth_hz=5.0,
        nominal_band_relative_halfwidth=0.05,
        hf_band_hz=(300.0, 450.0),
        max_epochs=1,
        patience=1,
        learning_rate=0.001,
        weight_decay=0.0,
        bootstrap_repetitions=100,
    )

    result = train_one_fold(
        "N1",
        outer_fold=0,
        seed=20260723,
        config=config,
        batch_size=2,
        validation_fraction=0.25,
        device=torch.device("cpu"),
    )

    predictions = pd.read_csv(result.oof_path)
    assert set(predictions["sample_id"]) == {"s0", "s3"}
    assert set(predictions["model"]) == {"N1"}
    assert result.checkpoint_path.is_file()
    assert result.log_path.is_file()
    metadata = json.loads(
        (result.checkpoint_path.parent / "run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["status"] == "complete"
    assert metadata["requested_max_epochs"] == 1
    assert metadata["batch_size"] == 2
    assert metadata["validation_fraction"] == 0.25
    assert metadata["fusion_encoder"] == "cnn"
    assert len(metadata["run_fingerprint"]) == 64
