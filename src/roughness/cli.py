import argparse
import json
import platform
import sys
from pathlib import Path

import sklearn
import torch

from .baselines import cross_validated_baselines
from .config import load_config
from .manifest import build_manifest
from .reporting import save_diagnostic_plots, summarize_metrics
from .splits import make_group_folds


def run(config_path: Path, output_override: Path | None = None) -> dict:
    config = load_config(config_path)
    output = output_override or config.output_dir
    output.mkdir(parents=True, exist_ok=True)

    manifest, audit = build_manifest(config)
    folds = make_group_folds(manifest, config.n_splits, config.seed)
    predictions, fold_metrics = cross_validated_baselines(
        manifest, folds, config.seed
    )
    summary = summarize_metrics(fold_metrics)
    best_model = save_diagnostic_plots(predictions, summary, output)

    manifest.to_csv(
        output / "manifest.csv", index=False, encoding="utf-8-sig"
    )
    folds.to_csv(output / "folds.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(
        output / "oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    fold_metrics.to_csv(
        output / "fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        output / "summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "seed": config.seed,
        "n_splits": config.n_splits,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "best_model_by_weighted_mae": best_model,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.config)


if __name__ == "__main__":
    main()
