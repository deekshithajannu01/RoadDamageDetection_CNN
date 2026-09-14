"""Training-curve and run-metadata helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import ultralytics  # noqa: E402


def plot_training_curves(results_csv: Path, output: Path) -> None:
    frame = pd.read_csv(results_csv)
    frame.columns = [column.strip() for column in frame.columns]
    epoch = frame.get("epoch", pd.Series(range(len(frame))))
    panels = [
        ("Localization loss", ["train/box_loss", "val/box_loss"]),
        ("Classification loss", ["train/cls_loss", "val/cls_loss"]),
        ("Distribution focal loss", ["train/dfl_loss", "val/dfl_loss"]),
        ("Validation metrics", ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, (title, columns) in zip(axes.flat, panels):
        found = False
        for column in columns:
            if column in frame:
                axis.plot(epoch, frame[column], label=column)
                found = True
        axis.set(title=title, xlabel="Epoch")
        if found:
            axis.legend()
        else:
            axis.text(0.5, 0.5, "Metric not emitted", ha="center", va="center")
    fig.suptitle("YOLOv8 training and validation curves")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_metadata(output: Path, settings: dict[str, object]) -> None:
    metadata = {
        "settings": settings,
        "git_commit": git_commit(),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    output.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
