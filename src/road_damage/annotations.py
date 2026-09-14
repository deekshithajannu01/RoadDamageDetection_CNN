"""Pascal VOC parsing, cleaning, and YOLO conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from road_damage.constants import CLASS_TO_ID


@dataclass(frozen=True)
class Box:
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


@dataclass(frozen=True)
class VocAnnotation:
    filename: str
    width: int
    height: int
    boxes: tuple[Box, ...]


def _number(element: ElementTree.Element | None, field: str, default: float = 0.0) -> float:
    if element is None:
        return default
    node = element.find(field)
    if node is None or node.text is None:
        return default
    return float(node.text)


def parse_voc(xml_path: Path) -> VocAnnotation:
    """Read one Pascal VOC XML annotation without silently changing boxes."""
    root = ElementTree.parse(xml_path).getroot()
    size = root.find("size")
    width = int(_number(size, "width"))
    height = int(_number(size, "height"))
    filename = root.findtext("filename", default=f"{xml_path.stem}.jpg")

    boxes: list[Box] = []
    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        bounds = obj.find("bndbox")
        if bounds is None:
            continue
        boxes.append(
            Box(
                class_name=class_name,
                xmin=_number(bounds, "xmin"),
                ymin=_number(bounds, "ymin"),
                xmax=_number(bounds, "xmax"),
                ymax=_number(bounds, "ymax"),
            )
        )
    return VocAnnotation(filename=filename, width=width, height=height, boxes=tuple(boxes))


def clean_box(box: Box, width: int, height: int, min_size_px: float = 2.0) -> Box | None:
    """Clamp a known-class box to its image and reject malformed/tiny boxes."""
    if box.class_name not in CLASS_TO_ID or width <= 0 or height <= 0:
        return None
    if box.xmax <= box.xmin or box.ymax <= box.ymin:
        return None
    cleaned = Box(
        class_name=box.class_name,
        xmin=max(0.0, min(float(width), box.xmin)),
        ymin=max(0.0, min(float(height), box.ymin)),
        xmax=max(0.0, min(float(width), box.xmax)),
        ymax=max(0.0, min(float(height), box.ymax)),
    )
    if cleaned.width < min_size_px or cleaned.height < min_size_px:
        return None
    return cleaned


def box_to_yolo(box: Box, width: int, height: int) -> tuple[int, float, float, float, float]:
    """Convert an in-bounds xyxy box to normalized YOLO class/xywh values."""
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    x_center = ((box.xmin + box.xmax) / 2.0) / width
    y_center = ((box.ymin + box.ymax) / 2.0) / height
    box_width = box.width / width
    box_height = box.height / height
    values = (x_center, y_center, box_width, box_height)
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Box is outside image after cleaning: {box}")
    return CLASS_TO_ID[box.class_name], *values


def format_yolo_line(box: Box, width: int, height: int) -> str:
    class_id, x_center, y_center, box_width, box_height = box_to_yolo(box, width, height)
    return f"{class_id} {x_center:.8f} {y_center:.8f} " f"{box_width:.8f} {box_height:.8f}"
