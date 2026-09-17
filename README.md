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


