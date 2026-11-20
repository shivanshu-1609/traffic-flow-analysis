"""Convert traffic annotations from Pascal VOC XML to YOLO format."""

import argparse
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from PIL import Image

CLASSES = (
    "ambulance",
    "army vehicle",
    "auto rickshaw",
    "bicycle",
    "bus",
    "car",
    "garbagevan",
    "human hauler",
    "minibus",
    "minivan",
    "motorbike",
    "pickup",
    "policecar",
    "rickshaw",
    "scooter",
    "suv",
    "taxi",
    "three wheelers (CNG)",
    "truck",
    "van",
    "wheelbarrow",
)


def yolo_box(bounds, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("Bounding box values must be finite.")
    xmin, ymin, xmax, ymax = bounds
    xmin, xmax = max(0, xmin), min(width, xmax)
    ymin, ymax = max(0, ymin), min(height, ymax)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid bounding box: {bounds}")
    return (
        (xmin + xmax) / (2 * width),
        (ymin + ymax) / (2 * height),
        (xmax - xmin) / width,
        (ymax - ymin) / height,
    )


def prepare(images, annotations, output, seed=42):
    records, seen = [], set()
    for path in sorted(annotations.glob("*.xml")):
        tree = ET.parse(path).getroot()
        filename = tree.findtext("filename")
        if not filename:
            raise ValueError(f"Missing image filename: {path.name}")
        source = (images / filename).resolve()
        if not source.is_relative_to(images.resolve()) or not source.is_file():
            raise ValueError(f"Image not found inside --images: {filename}")
        if source.stem in seen:
            raise ValueError(f"Duplicate image stem: {source.stem}")
        seen.add(source.stem)
        with Image.open(source) as image:
            width, height = image.size
        if (
            int(tree.findtext("size/width", "0")),
            int(tree.findtext("size/height", "0")),
        ) != (width, height):
            raise ValueError(f"XML dimensions differ from image: {filename}")
        lines = []
        for obj in tree.findall("object"):
            name = obj.findtext("name")
            if name not in CLASSES:
                raise ValueError(f"Unknown class {name!r} in {path.name}")
            bounds = [
                float(obj.findtext(f"bndbox/{key}"))
                for key in ("xmin", "ymin", "xmax", "ymax")
            ]
            values = yolo_box(bounds, width, height)
            lines.append(
                f"{CLASSES.index(name)} " + " ".join(f"{value:.8f}" for value in values)
            )
        records.append((source, lines))
    if len(records) < 2:
        raise ValueError("At least two XML/image pairs are required.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory.")
    random.Random(seed).shuffle(records)
    val_count = min(len(records) - 1, max(1, round(len(records) * 0.2)))
    for index, (source, lines) in enumerate(records):
        split = "val" if index < val_count else "train"
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image.convert("RGB").save(image_dir / f"{source.stem}.jpg")
        (label_dir / f"{source.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    configuration = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(CLASSES)),
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8"
    )
    print(
        f"Prepared {len(records) - val_count} training "
        f"and {val_count} validation images."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/yolo"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        prepare(args.images, args.annotations, args.output, args.seed)
    except (ValueError, FileNotFoundError, ET.ParseError) as error:
        parser.exit(2, f"Error: {error}\n")
