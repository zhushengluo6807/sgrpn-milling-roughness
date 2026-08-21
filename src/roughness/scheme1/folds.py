import pandas as pd


_KEY_COLUMNS = ["sample_id", "group_id"]


def validate_outer_folds(
    manifest: pd.DataFrame,
    folds: pd.DataFrame,
    expected_n_folds: int = 5,
) -> dict[str, int]:
    missing_manifest_columns = sorted(set(_KEY_COLUMNS) - set(manifest.columns))
    missing_fold_columns = sorted(
        set([*_KEY_COLUMNS, "fold"]) - set(folds.columns)
    )
    if missing_manifest_columns:
        raise ValueError(f"Manifest missing columns: {missing_manifest_columns}")
    if missing_fold_columns:
        raise ValueError(f"Folds missing columns: {missing_fold_columns}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Duplicate manifest sample_id values")
    if folds["sample_id"].duplicated().any():
        raise ValueError("Duplicate fold assignments")

    manifest_keys = set(manifest["sample_id"].astype(str))
    fold_keys = set(folds["sample_id"].astype(str))
    missing = sorted(manifest_keys - fold_keys)
    unknown = sorted(fold_keys - manifest_keys)
    if missing:
        raise ValueError(f"Missing fold assignments: {missing[:10]}")
    if unknown:
        raise ValueError(f"Unknown fold assignments: {unknown[:10]}")

    expected_groups = manifest.set_index("sample_id")["group_id"].astype(str)
    actual_groups = folds.set_index("sample_id")["group_id"].astype(str)
    actual_groups = actual_groups.reindex(expected_groups.index)
    mismatched = expected_groups[expected_groups != actual_groups]
    if not mismatched.empty:
        raise ValueError(
            f"Fold group_id does not match manifest: {mismatched.index[:10].tolist()}"
        )

    fold_values = pd.to_numeric(folds["fold"], errors="coerce")
    if fold_values.isna().any() or (fold_values % 1 != 0).any():
        raise ValueError("Fold values must be integers")
    unique_folds = sorted(fold_values.astype(int).unique().tolist())
    if unique_folds != list(range(expected_n_folds)):
        raise ValueError(
            f"Expected fold IDs 0..{expected_n_folds - 1}, got {unique_folds}"
        )

    assigned = folds.assign(
        group_id=folds["group_id"].astype(str),
        fold=fold_values.astype(int),
    )
    group_fold_counts = assigned.groupby("group_id")["fold"].nunique()
    overlap_count = int((group_fold_counts > 1).sum())
    if overlap_count:
        raise ValueError(f"Group leakage detected for {overlap_count} groups")

    return {
        "n_samples": int(len(manifest)),
        "n_groups": int(manifest["group_id"].astype(str).nunique()),
        "n_folds": int(len(unique_folds)),
        "group_overlap_count": overlap_count,
    }
