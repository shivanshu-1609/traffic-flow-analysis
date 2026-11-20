"""Train, validate, or run YOLOv8 traffic detection."""

import argparse
import csv
from pathlib import Path


def run(args):
    from ultralytics import YOLO

    if args.command == "train":
        if min(args.epochs, args.batch, args.image_size) < 1 or not args.data.is_file():
            raise ValueError("Provide data.yaml and positive training settings.")
        model = YOLO(args.weights)
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.image_size,
            batch=args.batch,
            workers=0,
            project=str(args.output),
            name="train",
            seed=42,
        )
    elif args.command == "validate":
        YOLO(args.weights).val(
            data=str(args.data), project=str(args.output), name="validation"
        )
    else:
        if not 0 <= args.confidence <= 1:
            raise ValueError("Confidence must lie between zero and one.")
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Choose an empty output directory.")
        source = int(args.source) if args.source.isdigit() else args.source
        model = YOLO(args.weights)
        method = model.track if args.track else model.predict
        options = dict(
            source=source,
            conf=args.confidence,
            stream=True,
            save=True,
            project=str(args.output),
            name="predictions",
            verbose=False,
        )
        if args.track:
            options.update(persist=True, tracker="bytetrack.yaml")
        results = method(**options)
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "detections.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "frame",
                    "source",
                    "class",
                    "confidence",
                    "track_id",
                    "xmin",
                    "ymin",
                    "xmax",
                    "ymax",
                ]
            )
            for frame, result in enumerate(results):
                boxes = result.boxes
                if boxes is None:
                    continue
                coordinates, confidences, labels = (
                    boxes.xyxy.cpu().tolist(),
                    boxes.conf.cpu().tolist(),
                    boxes.cls.cpu().tolist(),
                )
                ids = (
                    boxes.id.cpu().tolist()
                    if boxes.id is not None
                    else [None] * len(labels)
                )
                for box, confidence, label, track_id in zip(
                    coordinates, confidences, labels, ids, strict=True
                ):
                    writer.writerow(
                        [
                            frame,
                            result.path,
                            result.names[int(label)],
                            confidence,
                            int(track_id) if track_id is not None else "",
                            *box,
                        ]
                    )
        print(f"Saved detections to {args.output / 'detections.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "validate", "detect"):
        command = commands.add_parser(name)
        command.add_argument("--weights", default="yolov8n.pt")
        command.add_argument("--output", type=Path, default=Path(f"outputs/{name}"))
        if name in ("train", "validate"):
            command.add_argument("--data", type=Path, required=True)
        else:
            command.add_argument("--source", required=True)
            command.add_argument("--confidence", type=float, default=0.5)
            command.add_argument("--track", action="store_true")
        if name == "train":
            command.add_argument("--epochs", type=int, default=20)
            command.add_argument("--batch", type=int, default=8)
            command.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
