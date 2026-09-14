"""Screen-space severity and explicitly calibrated physical-size estimates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from road_damage.constants import PROJECT_ROOT


def load_severity_config(path: Path | None = None) -> dict[str, Any]:
    path = path or PROJECT_ROOT / "configs" / "severity.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Severity configuration must be a mapping: {path}")
    return config


def estimate_severity(
    xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    class_name: str,
    config: dict[str, Any],
    meters_per_pixel: float | None = None,
) -> dict[str, Any]:
    """Estimate bbox footprint, with physical units only when a scale is supplied."""
    xmin, ymin, xmax, ymax = xyxy
    width_px = max(0.0, xmax - xmin)
    height_px = max(0.0, ymax - ymin)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    area_ratio = (width_px * height_px) / (image_width * image_height)
    thresholds = config["area_ratio_thresholds"].get(
        class_name, config["area_ratio_thresholds"]["D40"]
    )
    labels = config.get("labels", ["minor", "moderate", "severe"])
    severity_index = 0 if area_ratio < thresholds[0] else 1 if area_ratio < thresholds[1] else 2
    result: dict[str, Any] = {
        "severity": labels[severity_index],
        "area_ratio": area_ratio,
        "bbox_width_px": width_px,
        "bbox_height_px": height_px,
        "physical_size": None,
        "method": "screen-space bbox-area heuristic",
    }
    scale = meters_per_pixel or config.get("default_meters_per_pixel")
    if scale is not None:
        scale = float(scale)
        if scale <= 0:
            raise ValueError("meters_per_pixel must be positive")
        uncertainty = float(config.get("relative_uncertainty", 0.25))
        width_m = width_px * scale
        height_m = height_px * scale
        result["physical_size"] = {
            "width_m": width_m,
            "height_m": height_m,
            "bbox_area_m2": width_m * height_m,
            "relative_uncertainty": uncertainty,
        }
        result["method"] = "single-scale calibrated bbox footprint"
    return result
