"""Tune confidence and NMS IoU thresholds on the validation split."""

from __future__ import annotations

import argparse
import csv
import itertools
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

from road_damage.constants import PROJECT_ROOT
from road_damage.metrics import extract_detection_metrics, scores_from_confusion


def _float_list(value: str) -> list[float]:
    try:
        values = [float(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated floats") from error
    if not values or any(item <= 0 or item >= 1 for item in values):
        raise argparse.ArgumentTypeError("Thresholds must be between 0 and 1")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "data" / "processed" / "rdd2022.yaml"
    )
    parser.add_argument("--confidence", type=_float_list, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--iou", type=_float_list, default=[0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "tuning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    rows: list[dict[str, float]] = []
    with tempfile.TemporaryDirectory(prefix="rdd-threshold-") as temporary:
        for confidence, iou in itertools.product(args.confidence, args.iou):
            result = model.val(
                data=str(args.data),
                split="val",
                conf=confidence,
                iou=iou,
                imgsz=args.imgsz,
                device=args.device,
                plots=True,
                project=temporary,
                name=f"conf_{confidence:.3f}_iou_{iou:.3f}",
                exist_ok=True,
                verbose=False,
            )
            metrics = extract_detection_metrics(result)
            confusion = getattr(result, "confusion_matrix", None)
            if confusion is None:
                confusion = getattr(model.validator, "confusion_matrix", None)
            if confusion is None:
                raise RuntimeError("Ultralytics did not expose its detection confusion matrix")
            operating = scores_from_confusion(confusion.matrix, len(metrics["per_class"]))
            rows.append(
                {
                    "confidence": confidence,
                    "iou": iou,
                    "macro_f1": operating["macro_f1"],
                    "map50": float(metrics["map50"]),
                    "map50_95": float(metrics["map50_95"]),
                    "precision": operating["precision"],
                    "recall": operating["recall"],
                }
            )
            print(f"conf={confidence:.2f} iou={iou:.2f} " f"macro-F1={operating['macro_f1']:.4f}")

    best = max(rows, key=lambda row: (row["macro_f1"], row["map50_95"]))
    with (args.output / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best_config = {
        "confidence": best["confidence"],
        "nms_iou": best["iou"],
        "selection_metric": "macro_f1",
        "validation_macro_f1": best["macro_f1"],
        "model": str(args.model.resolve()),
    }
    (args.output / "best_thresholds.yaml").write_text(
        yaml.safe_dump(best_config, sort_keys=False), encoding="utf-8"
    )
    print(f"Best validation thresholds: conf={best['confidence']}, IoU={best['iou']}")


if __name__ == "__main__":
    main()
