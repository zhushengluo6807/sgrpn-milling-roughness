"""Resumable command-line launcher for the frozen corrective analysis."""

from __future__ import annotations

import argparse
import hashlib
import json

import torch

from .config import load_phase_b_config, load_sgrpn_config
from .corrective import (
    load_corrective_config,
    run_corrective_all,
    run_corrective_fold,
)
from .corrective_evaluation import evaluate_corrective_once
from .data import load_data_bundle
from .order_spectrum import load_order_cache
from .phase_b_training import PHASE_B_SEEDS


def _selected_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return requested


def _load_context(config_path: str):
    corrective_config = load_corrective_config(config_path)
    phase_b_config = load_phase_b_config(corrective_config.phase_b_config_path)
    actual_phase_a_hash = hashlib.sha256(
        phase_b_config.phase_a_config_path.read_bytes()
    ).hexdigest()
    if actual_phase_a_hash != phase_b_config.phase_a_config_file_sha256:
        raise ValueError("Phase A config hash does not match the frozen Phase B registration")
    phase_a_config = load_sgrpn_config(phase_b_config.phase_a_config_path)
    bundle = load_data_bundle(phase_a_config)
    cache = load_order_cache(bundle, phase_a_config)
    return corrective_config, phase_b_config, phase_a_config, bundle, cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roughness-sgrpn-corrective")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--fold", type=int, choices=range(5))
    train.add_argument("--seed", type=int, choices=PHASE_B_SEEDS)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    return parser


def _message(**values: object) -> None:
    print(json.dumps(values, ensure_ascii=False, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train" and (args.fold is None) != (args.seed is None):
        parser.error("--fold and --seed must be supplied together")

    corrective_config, phase_b_config, phase_a_config, bundle, cache = _load_context(
        args.config
    )
    context = (
        corrective_config,
        phase_b_config,
        phase_a_config,
        bundle,
        cache,
    )
    if args.command == "evaluate":
        evaluate_corrective_once(
            corrective_config,
            phase_b_config,
            bundle,
            cache,
        )
        _message(evaluation_status="complete", evaluation_invocations=1)
        return 0
    device = _selected_device(args.device)
    if args.fold is not None:
        run_corrective_fold(
            *context,
            fold=args.fold,
            seed=args.seed,
            device=device,
        )
        _message(
            training_status="partial",
            fold=args.fold,
            seed=args.seed,
            device=device,
        )
        return 0

    completed_units = 0
    for fold in range(5):
        for seed in PHASE_B_SEEDS:
            run_corrective_fold(
                *context,
                fold=fold,
                seed=seed,
                device=device,
            )
            completed_units += 1
            _message(
                training_status="partial",
                completed_units=completed_units,
                total_units=15,
                fold=fold,
                seed=seed,
                device=device,
            )

    run_corrective_all(*context, device=device)
    _message(training_status="complete", completed_units=15, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
