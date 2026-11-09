"""Snapshot comparison helpers for tabular datasets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


def compare_snapshots(
    before: pd.DataFrame,
    after: pd.DataFrame,
    key_columns: Sequence[Any],
) -> dict[str, Any]:
    """Compare two tables by a unique, non-null record key.

    Values with missing data in both snapshots are considered unchanged. Rows
    with missing or repeated keys are rejected so the comparison cannot silently
    pair the wrong records.
    """
    keys = list(key_columns)
    if not keys:
        raise ValueError("Choose at least one column that uniquely identifies a record.")
    if before.columns.has_duplicates or after.columns.has_duplicates:
        raise ValueError("Both snapshots need unique column names before they can be compared.")
    missing_before = [column for column in keys if column not in before.columns]
    missing_after = [column for column in keys if column not in after.columns]
    if missing_before or missing_after:
        raise ValueError("Every selected key column must exist in both snapshots.")

    def keyed(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        null_rows = int(frame[keys].isna().any(axis=1).sum())
        if null_rows:
            raise ValueError(f"The {label} snapshot has {null_rows:,} row(s) with a missing record key.")
        duplicate_rows = int(frame.duplicated(subset=keys, keep=False).sum())
        if duplicate_rows:
            raise ValueError(
                f"The {label} snapshot has {duplicate_rows:,} row(s) with repeated record keys. "
                "Choose a more specific key or resolve duplicates first."
            )
        return frame.set_index(keys, drop=True)

    before_keyed = keyed(before, "current")
    after_keyed = keyed(after, "comparison")
    before_index = before_keyed.index
    after_index = after_keyed.index

    added_mask = ~after_index.isin(before_index)
    removed_mask = ~before_index.isin(after_index)
    added_rows = after_keyed.loc[added_mask].reset_index()
    removed_rows = before_keyed.loc[removed_mask].reset_index()

    common_columns = [
        column for column in before.columns
        if column in after.columns and column not in keys
    ]
    common_index = before_index[before_index.isin(after_index)]
    changed_parts: list[pd.DataFrame] = []
    for column in common_columns:
        old_values = before_keyed.loc[common_index, column]
        new_values = after_keyed.loc[common_index, column]
        equal = old_values.eq(new_values).fillna(False) | (old_values.isna() & new_values.isna())
        changed_mask = ~equal.to_numpy()
        if changed_mask.any():
            keys_frame = common_index[changed_mask].to_frame(index=False)
            keys_frame["Changed field"] = column
            keys_frame["Old value"] = old_values.to_numpy()[changed_mask]
            keys_frame["New value"] = new_values.to_numpy()[changed_mask]
            changed_parts.append(keys_frame)

    if changed_parts:
        changed_cells = pd.concat(changed_parts, ignore_index=True)
        changed_rows = int(changed_cells[keys].drop_duplicates().shape[0])
    else:
        changed_cells = pd.DataFrame(columns=[*keys, "Changed field", "Old value", "New value"])
        changed_rows = 0

    return {
        "added_rows": added_rows,
        "removed_rows": removed_rows,
        "changed_cells": changed_cells,
        "added_count": int(added_mask.sum()),
        "removed_count": int(removed_mask.sum()),
        "changed_rows_count": changed_rows,
        "changed_cells_count": int(len(changed_cells)),
        "added_columns": [column for column in after.columns if column not in before.columns],
        "removed_columns": [column for column in before.columns if column not in after.columns],
    }
