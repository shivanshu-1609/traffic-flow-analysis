# Traffic Flow Analysis

Traffic object detection with YOLOv8, including custom vehicle classes, Pascal VOC
to YOLO annotation conversion, validation, and optional tracking.

## Data

Use the authorized WQU traffic images and XML annotations. The converter expects
an image directory and matching Pascal VOC XML files with filenames, image sizes,
class names, and bounding boxes. It supports the 21 course vehicle categories,
including cars, buses, rickshaws, and trucks; the mapping is defined in `prepare.py`.

```bash
python prepare.py --images data_images/train/images --annotations data_images/train/annotations --output data/yolo
```

This creates YOLO images, labels, and `data/yolo/data.yaml` with a seeded 80/20 image
split. Related frames from the same video can leak across an image split; use
separate recordings for train and validation when source-video metadata is available.

## Setup and usage

Use Python 3.11 and a separate virtual environment.

```bash
python -m pip install -r requirements.txt
python traffic.py train --data data/yolo/data.yaml
python traffic.py validate --data data/yolo/data.yaml --weights outputs/train/train/weights/best.pt
python traffic.py detect --source data/traffic.mp4 --weights outputs/train/train/weights/best.pt
```

For detection with the standard pretrained COCO model:

```bash
python traffic.py detect --source data/traffic.mp4 --weights yolov8n.pt
```

Add `--track` for ByteTrack IDs. Standard COCO weights do not distinguish the
course's custom vehicle categories; those require the custom checkpoint.

Inference saves annotated media and `detections.csv` containing frame number,
class, confidence, tracking ID when available, and bounding-box coordinates.
Repeated detections across frames are not unique vehicle counts. Track IDs can
change after occlusion; the project does not estimate calibrated speed or congestion.

## Evaluation

Use validation precision, recall, mAP50, and mAP50–95 from the trained model.
The augmentation notebook records a one-epoch run with mAP50 0.414 and
mAP50–95 0.286 on 578 validation images. These are historical course outputs,
not a benchmark reproduced with the supplied scripts.

## References and license

[WorldQuant University](https://www.wqu.edu/) provided the course material and lab
data access. [Ultralytics documentation](https://docs.ultralytics.com/tasks/detect/)
describes the detector API. The standalone scripts use MIT; Ultralytics, pretrained
weights, and datasets retain their upstream licenses and terms.
