# AI-Powered Road Damage Detection — RDD2022 India

Detects and localizes road damage with object detection—not image classification—because a road-maintenance system needs to know where damage is, not only whether an image contains damage.

## Core Idea

Given a road image, the system predicts both the type of damage and its bounding-box location. A classifier could only label the whole image as damaged or undamaged; object detection provides the localization and class information needed to mark individual cracks and potholes for inspection.

## Dataset

This project uses the **India subset of RDD2022** from the official [Figshare archive](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547):

- approximately 502 MiB compressed and 625 MiB extracted;
- 7,706 annotated images used to build this project's train/validation/test splits;
- Pascal VOC XML annotations converted to normalized YOLO format;
- downloaded directly from the 13.3 GB archive with HTTP byte-range requests, without saving or transferring the other countries.

| YOLO ID | RDD2022 code | Damage type |
|---:|---|---|
| 0 | D00 | Longitudinal crack |
| 1 | D10 | Transverse crack |
| 2 | D20 | Alligator crack |
| 3 | D40 | Pothole |

India-only was chosen for more consistent road surfaces, camera conditions, and lighting, while keeping the dataset manageable enough for repeated experiments.

The full India subset is now downloaded and prepared into 5,385 training, 1,571 validation, and 750 held-out test images. The generated [EDA summary](results/eda/summary.md) and plots document class balance, object sizes, resolution, and lighting.

## Deterministic Sampling

For hardware-limited experiments, preparation can select a smaller random subset before splitting. Sampling is reproducible: the same image limit and seed select the same source images every time.

```bash
rdd-prepare --max-images 3000 --seed 42
```

This reduces preparation and training work but does not reduce the initial India download. A 50–100-image subset should be used only for pipeline checks, never for final reported results.

## Pipeline Overview

1. Prepare — [`src/road_damage/prepare.py`](src/road_damage/prepare.py) — `rdd-prepare`
2. EDA — [`src/road_damage/eda.py`](src/road_damage/eda.py) — `rdd-eda`
3. Train — [`src/road_damage/train.py`](src/road_damage/train.py) — `rdd-train`
4. Tune — [`src/road_damage/tune.py`](src/road_damage/tune.py) — `rdd-tune`
5. Evaluate — [`src/road_damage/evaluate.py`](src/road_damage/evaluate.py) — `rdd-evaluate`
6. Failure analysis — [`src/road_damage/failure_analysis.py`](src/road_damage/failure_analysis.py) — `rdd-failure-analysis`
7. Predict — [`src/road_damage/predict.py`](src/road_damage/predict.py) — `rdd-predict`

## Setup

Requirements:

- Python 3.8 or newer;
- at least 2 GB of free space for the India data, with additional space for checkpoints and reports;
- an NVIDIA GPU is recommended for training, but Ultralytics can run on CPU more slowly.

```bash
git clone <your-repository-url>
cd CNN_roaddamagedetector

python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -e .
```

In VS Code, select `.venv/bin/python` as the Python interpreter. The project does not require an API server, Docker, CI, or a test framework to run the model workflow.

## Usage

Run these commands from the repository root with the virtual environment activated:

```bash
# 1. Download and extract only India
rdd-download

# 2. Clean annotations, convert them to YOLO, and create grouped 70/20/10 splits
rdd-prepare

# Optional alternative: prepare a reproducible 3,000-image experiment
# rdd-prepare --max-images 3000 --seed 42

# 3. Generate EDA tables, plots, and a Markdown summary
rdd-eda

# 4. Verify the training pipeline; do not report this smoke run as a final result
rdd-train --smoke

# 5. Train YOLOv8s from COCO-pretrained weights
rdd-train

# 6. Tune confidence and NMS IoU thresholds on validation data
rdd-tune \
  --model results/training/yolov8s_rdd2022/weights/best.pt

# 7. Evaluate once on the held-out test split
rdd-evaluate \
  --model results/training/yolov8s_rdd2022/weights/best.pt

# 8. Explain the weakest class using evaluation and EDA evidence
rdd-failure-analysis

# 9. Predict on an unseen image
rdd-predict path/to/new_road_image.jpg \
  --model results/training/yolov8s_rdd2022/weights/best.pt
```

Training settings are reviewable in [`configs/train.yaml`](configs/train.yaml). The baseline uses YOLOv8s, 100 epochs with early stopping, batch size 16, 768-pixel input, AdamW with a 0.001 initial learning rate, and training-only mosaic, horizontal flip, scale/crop, and conservative brightness/contrast augmentation. Reduce the batch size to 8 or 4 if GPU memory is insufficient.

Generated files are written to:

- `data/processed/` — YOLO images, labels, split manifest, dataset YAML, and cleaning report;
- `results/eda/` — class balance, bounding-box, resolution, and lighting analysis;
- `results/training/` — checkpoints, loss curves, mAP curves, and run metadata;
- `results/tuning/` — threshold sweep and selected confidence/NMS values;
- `results/evaluation/` — metrics, confusion matrices, precision-recall plots, and failure analysis;
- `results/inference/` — annotated predictions and `detections.json`.

## Expected Output

No genuine model detection exists in this repository yet because the model has not been trained. The India dataset is downloaded, prepared, and analyzed. After training, `rdd-predict` saves an annotated image and a literal detection record to `results/inference/detections.json`; copy one unchanged record from that file here instead of inventing an example.

The generated record has these fields:

```text
class, confidence, xyxy bounding box, image-area ratio, severity label, and optional calibrated size
```

## Evaluation / Results

**Evaluation scope: held-out split created only from the RDD2022 India subset. These results must not be presented as full-RDD2022 leaderboard results.**

No trained checkpoint or `results/evaluation/metrics.json` file is currently present, so real metric values are not available yet.

| India held-out test metric | Result |
|---|---:|
| mAP@0.5 | Not measured |
| mAP@0.5:0.95 | Not measured |

| Class | Precision | Recall |
|---|---:|---:|
| D00 — longitudinal crack | Not measured | Not measured |
| D10 — transverse crack | Not measured | Not measured |
| D20 — alligator crack | Not measured | Not measured |
| D40 — pothole | Not measured | Not measured |

Run `rdd-evaluate` after training and copy the real values from `results/evaluation/summary.md`: mAP measures overall detection quality, while per-class precision shows how often a class prediction is correct and recall shows how much labeled damage the model finds.

## Repo Structure

```text
.
├── .gitignore
├── README.md
├── configs/
│   ├── severity.yaml
│   └── train.yaml
├── data/
│   ├── .gitkeep
│   └── README.md
├── pyproject.toml
├── results/
│   ├── README.md
│   ├── eda/
│   │   └── .gitkeep
│   ├── evaluation/
│   │   └── .gitkeep
│   ├── inference/
│   │   └── .gitkeep
│   ├── training/
│   │   └── .gitkeep
│   └── tuning/
│       └── .gitkeep
└── src/
    └── road_damage/
        ├── __init__.py
        ├── annotations.py
        ├── constants.py
        ├── download.py
        ├── eda.py
        ├── evaluate.py
        ├── failure_analysis.py
        ├── metrics.py
        ├── predict.py
        ├── prepare.py
        ├── reporting.py
        ├── severity.py
        ├── splitting.py
        ├── train.py
        └── tune.py
```

## Why This Matters

Road-maintenance teams often have more reported damage than they can inspect or repair immediately. A system that marks each crack or pothole in road images can help staff review routes faster, group repeated reports, and prioritize the places that need attention first, while leaving final repair decisions to qualified inspectors.

## Next Steps / Limitations

- Train the model and publish genuine India-only evaluation metrics and prediction examples.
- Inspect false positives and false negatives before claiming field readiness.
- Replace screen-space severity labels with measurements calibrated for camera height and perspective.
- Add an active-learning loop so reviewed mistakes can improve later training sets.
- Test multi-country data and held-out-country evaluation before claiming geographic generalization.
- Validate performance on locally collected road images under real traffic and weather conditions.
