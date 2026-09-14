"""Evaluate a tuned checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

from road_damage.constants import CLASS_DESCRIPTIONS, CLASS_NAMES, PROJECT_ROOT
from road_damage.metrics import extract_detection_metrics, per_class_scores_from_confusion


def _load_thresholds(path: Path | None) -> tuple[float, float]:
    if path is None or not path.exists():
        return 0.25, 0.7
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    return float(values["confidence"]), float(values["nms_iou"])


def _write_summary(metrics: dict[str, object], output: Path, confidence: float, iou: float) -> None:
    per_class = metrics["per_class"]
    rows = "\n".join(
        f"| {name} | {CLASS_DESCRIPTIONS[name]} | {values['precision']:.4f} | "
        f"{values['recall']:.4f} | {values['map50']:.4f} | {values['map50_95']:.4f} | "
        f"{values['support']} |"
        for name, values in per_class.items()
    )
    text = f"""# Held-out test evaluation

- Confidence threshold: {confidence:.3f}
- NMS IoU threshold: {iou:.3f}
- mAP@0.5: {metrics['map50']:.4f}
- mAP@0.5:0.95: {metrics['map50_95']:.4f}
- Mean precision: {metrics['mean_precision']:.4f}
- Mean recall: {metrics['mean_recall']:.4f}

| Class | Meaning | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Test boxes |
|---|---|---:|---:|---:|---:|---:|
{rows}

The confusion-matrix and precision/recall plots in this directory are generated directly by the
Ultralytics validator. Thresholds must be selected on validation data only; this test split is a
single final estimate and must not be used for further tuning.
"""
    (output / "summary.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "data" / "processed" / "rdd2022.yaml"
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=PROJECT_ROOT / "results" / "tuning" / "best_thresholds.yaml",
    )
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    tuned_conf, tuned_iou = _load_thresholds(args.thresholds)
    confidence = args.conf if args.conf is not None else tuned_conf
    iou = args.iou if args.iou is not None else tuned_iou
    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    options = {
        "data": str(args.data),
        "split": "test",
        "conf": confidence,
        "iou": iou,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "plots": True,
        "project": str(args.output.parent),
        "name": args.output.name,
        "exist_ok": True,
    }
    if args.device:
        options["device"] = args.device
    result = model.val(**options)
    metrics = extract_detection_metrics(result)
    confusion = getattr(result, "confusion_matrix", None)
    if confusion is None:
        confusion = getattr(model.validator, "confusion_matrix", None)
    if confusion is None:
        raise RuntimeError("Ultralytics did not expose its detection confusion matrix")
    precision, recall, f1 = per_class_scores_from_confusion(confusion.matrix, len(CLASS_NAMES))
    for index, name in enumerate(CLASS_NAMES):
        metrics["per_class"][name].update(
            {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
        )
    metrics["mean_precision"] = float(precision.mean())
    metrics["mean_recall"] = float(recall.mean())
    metrics.update(
        {
            "confidence_threshold": confidence,
            "nms_iou_threshold": iou,
            "model": str(args.model.resolve()),
            "data": str(args.data.resolve()),
        }
    )
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_summary(metrics, args.output, confidence, iou)
    print(json.dumps(metrics, indent=2))
    print(f"Evaluation report written to {args.output}")


if __name__ == "__main__":
    main()
