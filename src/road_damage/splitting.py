"""Leakage-aware multilabel splitting utilities."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from road_damage.constants import CLASS_NAMES, COUNTRIES


def sequence_group(country: str, stem: str, block_size: int = 10) -> str:
    """Keep nearby numbered frames together to reduce near-duplicate leakage."""
    match = re.search(r"(\d+)$", stem)
    if match is None or block_size <= 1:
        return f"{country}:{stem}"
    number = int(match.group(1))
    prefix = stem[: match.start(1)]
    return f"{country}:{prefix}{number // block_size:08d}"


def _split_indices(y: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_indices, test_indices = next(splitter.split(indices, y))
    return train_indices, test_indices


def stratified_group_split(
    records: Iterable[Mapping[str, object]],
    train_fraction: float = 0.7,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, str]:
    """Assign record keys to train/val/test using group-level multilabel strata.

    Each record needs ``key``, ``group``, ``country``, and ``classes`` fields.
    Damage classes and country are both represented in the stratification target.
    """
    fractions = np.array([train_fraction, val_fraction, test_fraction], dtype=float)
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train/val/test fractions must be positive and sum to 1")

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["group"])].append(record)
    group_names = sorted(grouped)
    if len(group_names) < 10:
        raise ValueError("At least 10 frame groups are required for a 70/20/10 split")

    features: list[list[int]] = []
    for group_name in group_names:
        items = grouped[group_name]
        classes = {str(label) for item in items for label in item["classes"]}  # type: ignore[union-attr]
        countries = {str(item["country"]) for item in items}
        features.append(
            [int(name in classes) for name in CLASS_NAMES]
            + [int(country in countries) for country in COUNTRIES]
        )
    y = np.asarray(features, dtype=np.int8)

    remaining_idx, test_idx = _split_indices(y, test_fraction, seed)
    relative_val = val_fraction / (train_fraction + val_fraction)
    train_local, val_local = _split_indices(y[remaining_idx], relative_val, seed + 1)
    train_idx = remaining_idx[train_local]
    val_idx = remaining_idx[val_local]

    group_split = {
        **{group_names[index]: "train" for index in train_idx},
        **{group_names[index]: "val" for index in val_idx},
        **{group_names[index]: "test" for index in test_idx},
    }
    return {
        str(item["key"]): group_split[group_name]
        for group_name, items in grouped.items()
        for item in items
    }
