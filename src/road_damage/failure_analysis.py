"""Explain the weakest class using test metrics and EDA evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from road_damage.constants import CLASS_DESCRIPTIONS, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "results" / "evaluation" / "metrics.json",
    )
    parser.add_argument(
        "--boxes",
        type=Path,
        default=PROJECT_ROOT / "results" / "eda" / "bbox_stats.csv",
    )
    parser.add_argument(
        "--lighting",
        type=Path,
        default=PROJECT_ROOT / "results" / "eda" / "lighting_stats.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "evaluation" / "failure_analysis.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (args.metrics, args.boxes):
        if not required.exists():
            raise FileNotFoundError(required)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    boxes = pd.read_csv(args.boxes)
    per_class = metrics["per_class"]
    worst = min(per_class, key=lambda name: per_class[name]["map50_95"])
    worst_metrics = per_class[worst]
    multiple_subsets = boxes["country"].nunique() > 1

    class_counts = boxes["class"].value_counts()
    worst_boxes = boxes[boxes["class"] == worst]
    overall_median_area = boxes["area_ratio"].median()
    class_median_area = worst_boxes["area_ratio"].median()
    overall_short_side = boxes[["width_norm", "height_norm"]].min(axis=1).median()
    class_short_side = worst_boxes[["width_norm", "height_norm"]].min(axis=1).median()
    reasons: list[str] = []
    if class_counts[worst] <= class_counts.quantile(0.25):
        reasons.append(
            f"class imbalance: {worst} has {class_counts[worst]:,} boxes, in the bottom quartile"
        )
    if class_median_area < 0.75 * overall_median_area:
        reasons.append(
            f"small targets: its median box area is {class_median_area * 100:.3f}% versus "
            f"{overall_median_area * 100:.3f}% overall"
        )
    if class_short_side < 0.75 * overall_short_side:
        reasons.append(
            "thin geometry: the shorter normalized box side is unusually small"
        )

    lighting_note = "Lighting evidence was unavailable."
    if args.lighting.exists():
        lighting = pd.read_csv(args.lighting)
        joined = (
            worst_boxes[["key"]]
            .drop_duplicates()
            .merge(lighting, on="key", how="inner")
        )
        if not joined.empty:
            lighting_note = (
                f"Images containing {worst} have brightness mean/std "
                f"{joined['brightness_mean'].mean():.1f}/{joined['brightness_mean'].std():.1f}."
            )
            if (
                joined["brightness_mean"].std()
                > lighting["brightness_mean"].std() * 1.15
            ):
                reasons.append(
                    "lighting variance: this class occurs across unusually varied brightness"
                )
    if not reasons:
        additional_hypothesis = (
            " and country-specific domain shift" if multiple_subsets else ""
        )
        reasons.append(
            "the measured size and frequency are not exceptional; inspect confusion-matrix pairs, "
            f"label ambiguity, and texture{additional_hypothesis}"
        )
    reason_lines = "\n".join(f"- {reason}" for reason in reasons)

    recommendations = [
        "Review false negatives and high-confidence false positives from the validator plots.",
        (
            "Report results by source subset before changing augmentation or oversampling."
            if multiple_subsets
            else "Compare errors across brightness, object-size, and road-surface groups."
        ),
        (
            "If the class is rare, try class-aware sampling; if it is small/thin, compare "
            "960 px input."
        ),
        (
            "Retrain and accept a change only when it improves validation results without "
            "touching test data."
        ),
    ]
    recommendation_lines = "\n".join(f"- {item}" for item in recommendations)
    report = f"""# Failure analysis

## Weakest class

**{worst} — {CLASS_DESCRIPTIONS[worst]}** has the lowest test mAP@0.5:0.95
({worst_metrics['map50_95']:.4f}); precision is {worst_metrics['precision']:.4f} and recall is
{worst_metrics['recall']:.4f}.

## Evidence-backed hypotheses

{reason_lines}

{lighting_note} These are correlations that guide inspection, not proof of causality.

## Next experiments

{recommendation_lines}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Failure analysis written to {args.output}")


if __name__ == "__main__":
    main()
