import numpy as np
import pandas as pd


def make_group_folds(
    manifest: pd.DataFrame, n_splits: int, seed: int
) -> pd.DataFrame:
    groups = np.array(sorted(manifest["group_id"].astype(str).unique()))
    if len(groups) < n_splits:
        raise ValueError("Number of groups is smaller than n_splits")

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    mapping = {
        group: fold
        for fold, chunk in enumerate(np.array_split(groups, n_splits))
        for group in chunk
    }
    result = manifest[["sample_id", "group_id"]].copy()
    result["fold"] = result["group_id"].astype(str).map(mapping).astype(int)
    if result.groupby("group_id")["fold"].nunique().max() != 1:
        raise AssertionError("Group leakage detected")
    return result.sort_values("sample_id").reset_index(drop=True)
