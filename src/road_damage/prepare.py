"""Clean RDD2022 Pascal VOC labels, split them, and create a YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import yaml
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from road_damage.annotations import Box, clean_box, format_yolo_line, parse_voc
from road_damage.constants import (
    CLASS_NAMES,
    COUNTRIES,
    DEFAULT_COUNTRIES,
    PROJECT_ROOT,
)
from road_damage.splitting import sequence_group, stratified_group_split


@dataclass(frozen=True)
class ImageRecord:
    key: str
    group: str
    country: str
    source_image: Path
    source_xml: Path
    width: int
    height: int
    boxes: tuple[Box, ...]

    @property
    def classes(self) -> set[str]:
        return {box.class_name for box in self.boxes}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _annotation_dir(raw_root: Path, country: str) -> Path | None:
    candidates = sorted(raw_root.glob(f"**/{country}/train/annotations/xmls"))
    return candidates[0] if candidates else None


def _image_for_xml(xml_path: Path, filename: str) -> Path | None:
    image_dir = xml_path.parents[2] / "images"
    named = image_dir / Path(filename).name
    if named.is_file():
        return named
    for extension in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        candidate = image_dir / f"{xml_path.stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def collect_records(
    raw_root: Path,
    countries: list[str],
    min_box_size: float,
    frame_group_size: int,
    max_images: int | None = None,
    seed: int = 42,
) -> tuple[list[ImageRecord], dict[str, int]]:
    report = {
        "xml_files": 0,
        "images_kept": 0,
        "images_missing": 0,
        "images_unreadable": 0,
        "boxes_seen": 0,
        "boxes_kept": 0,
        "boxes_unknown_class": 0,
        "boxes_invalid_or_tiny": 0,
        "dimension_mismatches": 0,
    }
    records: list[ImageRecord] = []
    for country in countries:
        annotation_dir = _annotation_dir(raw_root, country)
        if annotation_dir is None:
            print(f"Warning: no annotations found for {country}")
            continue
        xml_paths = sorted(annotation_dir.glob("*.xml"))
        if max_images is not None and len(xml_paths) > max_images:
            subset_seed = seed + COUNTRIES.index(country)
            xml_paths = sorted(random.Random(subset_seed).sample(xml_paths, max_images))
        for xml_path in tqdm(xml_paths, desc=f"Reading {country}", unit="image"):
            report["xml_files"] += 1
            try:
                annotation = parse_voc(xml_path)
            except (OSError, ValueError, KeyError, ElementTree.ParseError) as error:
                print(f"Warning: could not parse {xml_path}: {error}")
                report["images_unreadable"] += 1
                continue
            image_path = _image_for_xml(xml_path, annotation.filename)
            if image_path is None:
                report["images_missing"] += 1
                continue
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    image.verify()
            except (OSError, UnidentifiedImageError):
                report["images_unreadable"] += 1
                continue
            if annotation.width != width or annotation.height != height:
                report["dimension_mismatches"] += 1

            cleaned: list[Box] = []
            for box in annotation.boxes:
                report["boxes_seen"] += 1
                if box.class_name not in CLASS_NAMES:
                    report["boxes_unknown_class"] += 1
                    continue
                valid_box = clean_box(box, width, height, min_box_size)
                if valid_box is None:
                    report["boxes_invalid_or_tiny"] += 1
                    continue
                cleaned.append(valid_box)
                report["boxes_kept"] += 1

            key = f"{country}__{image_path.stem}"
            records.append(
                ImageRecord(
                    key=key,
                    group=sequence_group(country, image_path.stem, frame_group_size),
                    country=country,
                    source_image=image_path,
                    source_xml=xml_path,
                    width=width,
                    height=height,
                    boxes=tuple(cleaned),
                )
            )
            report["images_kept"] += 1
    return records, report


def _materialize(source: Path, destination: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "symlink":
        destination.symlink_to(source.resolve())
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _reset_output(output: Path, overwrite: bool) -> None:
    if not output.exists():
        return
    has_content = any(output.iterdir())
    if has_content and not overwrite:
        raise FileExistsError(f"{output} is not empty; pass --overwrite to rebuild it")
    if has_content:
        for directory in (output / "images", output / "labels"):
            if directory.is_dir():
                shutil.rmtree(directory)
        for filename in ("manifest.csv", "rdd2022.yaml", "cleaning_report.json"):
            target = output / filename
            if target.is_file() or target.is_symlink():
                target.unlink()


def write_dataset(
    records: list[ImageRecord],
    assignments: dict[str, str],
    output: Path,
    mode: str,
    report: dict[str, int],
    overwrite: bool,
) -> None:
    _reset_output(output, overwrite)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    for record in tqdm(records, desc="Writing YOLO dataset", unit="image"):
        split = assignments[record.key]
        extension = record.source_image.suffix.lower()
        image_name = f"{record.key}{extension}"
        image_target = output / "images" / split / image_name
        label_target = output / "labels" / split / f"{record.key}.txt"
        _materialize(record.source_image, image_target, mode)
        label_target.write_text(
            "\n".join(
                format_yolo_line(box, record.width, record.height)
                for box in record.boxes
            )
            + ("\n" if record.boxes else ""),
            encoding="utf-8",
        )
        counts = {
            name: sum(box.class_name == name for box in record.boxes)
            for name in CLASS_NAMES
        }
        manifest_rows.append(
            {
                "key": record.key,
                "group": record.group,
                "country": record.country,
                "split": split,
                "source_image": str(record.source_image.resolve()),
                "image_path": str(image_target.resolve()),
                "label_path": str(label_target.resolve()),
                "width": record.width,
                "height": record.height,
                "box_count": len(record.boxes),
                "classes": "|".join(sorted(record.classes)),
                **{f"count_{name}": counts[name] for name in CLASS_NAMES},
            }
        )

    fieldnames = list(manifest_rows[0])
    with (output / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    dataset_yaml = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (output / "rdd2022.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
    )
    split_counts = {
        split: sum(row["split"] == split for row in manifest_rows)
        for split in ("train", "val", "test")
    }
    report = {**report, "split_images": split_counts}
    (output / "cleaning_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"YOLO dataset written to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--countries", nargs="+", choices=COUNTRIES, default=list(DEFAULT_COUNTRIES)
    )
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-box-size", type=float, default=2.0)
    parser.add_argument(
        "--frame-group-size",
        type=int,
        default=10,
        help="Keep blocks of adjacent numbered frames in one split; use 1 to disable",
    )
    parser.add_argument(
        "--mode", choices=("hardlink", "copy", "symlink"), default="hardlink"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-images",
        type=positive_int,
        help="Deterministic random image limit per subset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, report = collect_records(
        args.raw,
        args.countries,
        args.min_box_size,
        args.frame_group_size,
        args.max_images,
        args.seed,
    )
    if not records:
        raise FileNotFoundError(
            f"No annotated RDD2022 images found under {args.raw}. Run rdd-download first."
        )
    random.Random(args.seed).shuffle(records)
    split_input = [
        {
            "key": record.key,
            "group": record.group,
            "country": record.country,
            "classes": record.classes,
        }
        for record in records
    ]
    assignments = stratified_group_split(
        split_input, args.train, args.val, args.test, args.seed
    )
    write_dataset(records, assignments, args.output, args.mode, report, args.overwrite)


if __name__ == "__main__":
    main()
