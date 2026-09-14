"""Run image/video inference and emit detections with severity estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from road_damage.constants import CLASS_NAMES, PROJECT_ROOT
from road_damage.severity import estimate_severity, load_severity_config


def result_records(
    result: Any, severity_config: dict[str, Any], meters_per_pixel: float | None
) -> list[dict[str, Any]]:
    height, width = result.orig_shape
    records: list[dict[str, Any]] = []
    if result.boxes is None:
        return records
    for box in result.boxes:
        xyxy = tuple(float(value) for value in box.xyxy[0].cpu().tolist())
        class_id = int(box.cls[0].item())
        fallback = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
        class_name = result.names.get(class_id, fallback)
        records.append(
            {
                "class_id": class_id,
                "class": class_name,
                "confidence": float(box.conf[0].item()),
                "xyxy": list(xyxy),
                **estimate_severity(
                    xyxy, width, height, class_name, severity_config, meters_per_pixel
                ),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Image, directory, video, URL, or camera index")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device")
    parser.add_argument("--meters-per-pixel", type=float)
    parser.add_argument("--reference-pixels", type=float)
    parser.add_argument("--reference-meters", type=float)
    parser.add_argument(
        "--severity-config", type=Path, default=PROJECT_ROOT / "configs" / "severity.yaml"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if (args.reference_pixels is None) != (args.reference_meters is None):
        raise ValueError("Provide both --reference-pixels and --reference-meters")
    meters_per_pixel = args.meters_per_pixel
    if args.reference_pixels is not None:
        if args.reference_pixels <= 0 or args.reference_meters <= 0:
            raise ValueError("Reference measurements must be positive")
        meters_per_pixel = args.reference_meters / args.reference_pixels

    severity_config = load_severity_config(args.severity_config)
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    options = {
        "source": args.source,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "save": True,
        "project": str(args.output.parent),
        "name": args.output.name,
        "exist_ok": True,
        "stream": True,
    }
    if args.device:
        options["device"] = args.device
    payload: list[dict[str, Any]] = []
    for frame_index, result in enumerate(model.predict(**options)):
        payload.append(
            {
                "source": str(result.path),
                "frame_index": frame_index,
                "image_size": {"height": result.orig_shape[0], "width": result.orig_shape[1]},
                "detections": result_records(result, severity_config, meters_per_pixel),
            }
        )
    (args.output / "detections.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Predictions and JSON written to {args.output}")


if __name__ == "__main__":
    main()
