"""Generate reproducible EDA tables, figures, and a Markdown summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from tqdm import tqdm  # noqa: E402

from road_damage.constants import CLASS_DESCRIPTIONS, CLASS_NAMES, PROJECT_ROOT


def read_box_table(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in tqdm(
        manifest.itertuples(index=False), total=len(manifest), desc="Reading boxes"
    ):
        label_path = Path(item.label_path)
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, values)
            index = int(class_id)
            if index < 0 or index >= len(CLASS_NAMES):
                continue
            rows.append(
                {
                    "key": item.key,
                    "country": item.country,
                    "split": item.split,
                    "class_id": index,
                    "class": CLASS_NAMES[index],
                    "x_center": x_center,
                    "y_center": y_center,
                    "width_norm": width,
                    "height_norm": height,
                    "area_ratio": width * height,
                    "aspect_ratio": width / height if height else float("nan"),
                    "width_px": width * item.width,
                    "height_px": height * item.height,
                }
            )
    return pd.DataFrame(rows)


def read_lighting_table(
    manifest: pd.DataFrame, max_images: int | None, seed: int
) -> pd.DataFrame:
    sample = manifest
    if max_images is not None and len(manifest) > max_images:
        sample = manifest.sample(max_images, random_state=seed)
    rows: list[dict[str, object]] = []
    for item in tqdm(
        sample.itertuples(index=False), total=len(sample), desc="Measuring lighting"
    ):
        image = cv2.imread(str(item.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        rows.append(
            {
                "key": item.key,
                "country": item.country,
                "split": item.split,
                "brightness_mean": float(image.mean()),
                "contrast_std": float(image.std()),
                "dark_pixel_fraction": float((image < 40).mean()),
                "bright_pixel_fraction": float((image > 215).mean()),
            }
        )
    return pd.DataFrame(rows)


def _save_plots(
    manifest: pd.DataFrame, boxes: pd.DataFrame, lighting: pd.DataFrame, output: Path
) -> None:
    sns.set_theme(style="whitegrid")

    counts = boxes["class"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=counts.index, y=counts.values, ax=axis, hue=counts.index, legend=False
    )
    axis.set(
        title="RDD2022 class distribution", xlabel="Class", ylabel="Bounding boxes"
    )
    for container in axis.containers:
        axis.bar_label(container, fmt="%d")
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_boxes = boxes.assign(area_percent=boxes["area_ratio"] * 100)
    sns.histplot(
        data=plot_boxes,
        x="area_percent",
        hue="class",
        bins=60,
        log_scale=True,
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[0],
    )
    axes[0].set(
        title="Normalized bounding-box area", xlabel="Image area (%) — log scale"
    )
    sns.scatterplot(
        data=boxes.sample(min(10000, len(boxes)), random_state=42),
        x="width_norm",
        y="height_norm",
        hue="class",
        alpha=0.35,
        s=15,
        ax=axes[1],
    )
    axes[1].set(
        title="Bounding-box shape",
        xlabel="Normalized width",
        ylabel="Normalized height",
    )
    fig.tight_layout()
    fig.savefig(output / "bbox_distribution.png", dpi=180)
    plt.close(fig)

    resolutions = (
        manifest.groupby(["width", "height"]).size().reset_index(name="images")
    )
    fig, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=resolutions,
        x="width",
        y="height",
        size="images",
        hue="images",
        sizes=(30, 600),
        ax=axis,
    )
    axis.set(
        title="Image resolution variance", xlabel="Width (px)", ylabel="Height (px)"
    )
    fig.tight_layout()
    fig.savefig(output / "image_resolutions.png", dpi=180)
    plt.close(fig)

    if not lighting.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.boxplot(data=lighting, x="country", y="brightness_mean", ax=axes[0])
        sns.boxplot(data=lighting, x="country", y="contrast_std", ax=axes[1])
        for axis in axes:
            axis.tick_params(axis="x", rotation=35)
        axes[0].set(title="Lighting by subset", ylabel="Mean grayscale brightness")
        axes[1].set(title="Contrast by subset", ylabel="Grayscale standard deviation")
        fig.tight_layout()
        fig.savefig(output / "lighting_variance.png", dpi=180)
        plt.close(fig)

    country_counts = (
        boxes.groupby(["country", "class"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CLASS_NAMES)
    )
    fig, axis = plt.subplots(figsize=(10, 6))
    sns.heatmap(country_counts, annot=True, fmt="d", cmap="Blues", ax=axis)
    axis.set(
        title="Class distribution by source subset", xlabel="Class", ylabel="Subset"
    )
    fig.tight_layout()
    fig.savefig(output / "class_by_country.png", dpi=180)
    plt.close(fig)


def _write_summary(
    manifest: pd.DataFrame, boxes: pd.DataFrame, lighting: pd.DataFrame, output: Path
) -> None:
    counts = boxes["class"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    nonzero = counts[counts > 0]
    imbalance = float(nonzero.max() / nonzero.min()) if len(nonzero) else float("nan")
    area = boxes["area_ratio"]
    empty_images = int((manifest["box_count"] == 0).sum())
    resolution_count = manifest[["width", "height"]].drop_duplicates().shape[0]
    subset_count = manifest["country"].nunique()
    stratification_target = "damage presence"
    if subset_count > 1:
        stratification_target += " and source subset"
    lighting_text = "Lighting statistics were not calculated."
    if not lighting.empty:
        lighting_text = (
            f"Mean grayscale brightness spans {lighting['brightness_mean'].min():.1f}–"
            f"{lighting['brightness_mean'].max():.1f}; the across-image standard deviation is "
            f"{lighting['brightness_mean'].std():.1f}."
        )

    class_lines = "\n".join(
        f"- {name} ({CLASS_DESCRIPTIONS[name]}): {int(counts[name]):,} boxes"
        for name in CLASS_NAMES
    )
    text = f"""# RDD2022 exploratory data analysis

This report is generated from the prepared split, not from hard-coded dataset statistics.

## Coverage and balance

- Images: {len(manifest):,} ({empty_images:,} have no retained target boxes)
- Bounding boxes: {len(boxes):,}
- Source subsets: {subset_count}
- Distinct image resolutions: {resolution_count:,}
- Largest/smallest non-empty class ratio: {imbalance:.2f}×

{class_lines}

## Object scale

- Median box area: {area.median() * 100:.3f}% of the image
- 10th–90th percentile: {area.quantile(0.1) * 100:.3f}%–{area.quantile(0.9) * 100:.3f}%
- Small objects (<1% of image area): {(area < 0.01).mean() * 100:.1f}%

Small and thin crack boxes motivate 768 px training and mosaic/scale augmentation. Very small
boxes are retained unless either dimension is below the cleaning threshold.

## Lighting

{lighting_text} Brightness/contrast jitter is therefore used conservatively so it improves
illumination robustness without erasing faint cracks.

## Leakage and split policy

The 70/20/10 split is multilabel-stratified on {stratification_target}. Consecutive
numbered frames are grouped in blocks (10 by default), preventing near-duplicate frames from
appearing on both sides of an evaluation boundary. See the CSV tables and figures in this folder
for the auditable source values.
"""
    (output / "summary.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results" / "eda"
    )
    parser.add_argument(
        "--lighting-sample",
        type=int,
        help="Optional deterministic image limit; default measures every image",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.dataset / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}; run rdd-prepare first")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    boxes = read_box_table(manifest)
    if boxes.empty:
        raise ValueError("No valid YOLO boxes were found in the prepared dataset")
    lighting = read_lighting_table(manifest, args.lighting_sample, args.seed)

    boxes.to_csv(args.output / "bbox_stats.csv", index=False)
    lighting.to_csv(args.output / "lighting_stats.csv", index=False)
    manifest.groupby(["split", "country"], as_index=False).size().to_csv(
        args.output / "image_counts.csv", index=False
    )
    boxes.groupby(["split", "country", "class"], as_index=False).size().to_csv(
        args.output / "class_counts.csv", index=False
    )
    _save_plots(manifest, boxes, lighting, args.output)
    _write_summary(manifest, boxes, lighting, args.output)
    print(f"EDA report written to {args.output}")


if __name__ == "__main__":
    main()
