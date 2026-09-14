"""Fine-tune COCO-pretrained YOLOv8 and save overfitting diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

from road_damage.constants import PROJECT_ROOT
from road_damage.reporting import plot_training_curves, write_run_metadata


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(config_path: Path) -> dict[str, object]:
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError(f"Training configuration must be a mapping: {config_path}")
    if "data" in settings:
        settings["data"] = str(_project_path(str(settings["data"])))
    if "project" in settings:
        settings["project"] = str(_project_path(str(settings["project"])))
    return settings


def build_custom_augmentations(specifications: object) -> list[object]:
    """Build a small, reviewed whitelist of bbox-safe photometric transforms."""
    if not specifications:
        return []
    if not isinstance(specifications, list):
        raise ValueError("custom_augmentations must be a list")
    import albumentations as albumentations

    allowed = {"RandomBrightnessContrast", "RandomGamma", "CLAHE"}
    transforms = []
    for specification in specifications:
        if not isinstance(specification, dict) or "name" not in specification:
            raise ValueError("Each custom augmentation needs a name")
        values = dict(specification)
        name = str(values.pop("name"))
        if name not in allowed:
            raise ValueError(f"Unsupported custom augmentation: {name}")
        transforms.append(getattr(albumentations, name)(**values))
    return transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "train.yaml")
    parser.add_argument("--device", help="CUDA device such as 0, 0,1, cpu, or mps")
    parser.add_argument("--resume", type=Path, help="Resume an interrupted Ultralytics last.pt run")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One epoch on 1%% of data at 320 px to verify plumbing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    model_name = str(settings.pop("model", "yolov8s.pt"))
    augmentations = build_custom_augmentations(settings.pop("custom_augmentations", None))
    if augmentations:
        settings["augmentations"] = augmentations
    if args.device:
        settings["device"] = args.device
    if args.smoke:
        settings.update(
            {
                "epochs": 1,
                "imgsz": 320,
                "batch": 2,
                "workers": 0,
                "fraction": 0.01,
                "name": f"{settings.get('name', 'yolov8s_rdd2022')}_smoke",
            }
        )

    if args.resume:
        model = YOLO(str(args.resume))
        model.train(resume=True)
    else:
        data_path = Path(str(settings["data"]))
        if not data_path.exists():
            raise FileNotFoundError(f"Missing {data_path}; run rdd-prepare first")
        model = YOLO(model_name)
        model.train(**settings)

    if getattr(model, "trainer", None) is None:
        raise RuntimeError("Ultralytics did not expose the completed trainer state")
    save_dir = Path(model.trainer.save_dir)
    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        plot_training_curves(results_csv, save_dir / "loss_and_metric_curves.png")
    write_run_metadata(save_dir / "run_metadata.json", settings)
    print(f"Training outputs: {save_dir}")
    print(f"Best checkpoint: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
