"""Create class-specific test metrics reports for the helmet detector.

This script evaluates the latest trained model on the held-out test split.
It saves Ultralytics evaluation plots plus CSV and text reports containing
separate metrics for the ``helmet`` and ``no_helmet`` classes.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "construction_helmet_yolo" / "data.yaml"
MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "construction_helmet_yolo26nExtra"
    / "weights"
    / "best.pt"
)
OUTPUT_PROJECT = PROJECT_ROOT / "runs" / "evaluation"
RUN_NAME = "test_metrics_class_report"
REPORT_DIRECTORY = OUTPUT_PROJECT / RUN_NAME
CSV_REPORT_PATH = REPORT_DIRECTORY / "class_metrics.csv"
TEXT_REPORT_PATH = REPORT_DIRECTORY / "class_metrics.txt"

IMAGE_SIZE = 416
BATCH_SIZE = 8
DEVICE: int | str = 0 if torch.cuda.is_available() else "cpu"

CSV_FIELDS = [
    "class_id",
    "class_name",
    "images",
    "instances",
    "precision",
    "recall",
    "f1",
    "mAP50",
    "mAP50-95",
]


def validate_inputs() -> None:
    """Stop with a clear error if a required project input is missing."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")


def collect_class_metrics(metrics: Any) -> list[dict[str, Any]]:
    """Convert Ultralytics per-class metrics into CSV-friendly rows."""
    rows: list[dict[str, Any]] = []

    # metric_index addresses the compact metric arrays. class_id is the real
    # dataset class ID, so this also works if a future test set omits a class.
    for metric_index, class_id_value in enumerate(metrics.ap_class_index):
        class_id = int(class_id_value)
        precision, recall, map50, map50_95 = metrics.class_result(metric_index)

        rows.append(
            {
                "class_id": class_id,
                "class_name": metrics.names[class_id],
                "images": int(metrics.nt_per_image[class_id]),
                "instances": int(metrics.nt_per_class[class_id]),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(metrics.box.f1[metric_index]),
                "mAP50": float(map50),
                "mAP50-95": float(map50_95),
            }
        )

    return rows


def save_csv_report(rows: list[dict[str, Any]]) -> None:
    """Save one machine-readable row for every evaluated class."""
    with CSV_REPORT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_text_report(metrics: Any, rows: list[dict[str, Any]]) -> None:
    """Save an easy-to-read overall and per-class evaluation report."""
    lines = [
        "Construction Helmet Detector - Held-out Test Report",
        "=" * 55,
        f"Model: {MODEL_PATH}",
        f"Dataset: {DATA_YAML}",
        f"Device: {DEVICE}",
        "",
        "Overall metrics",
        f"Precision: {metrics.box.mp:.5f}",
        f"Recall: {metrics.box.mr:.5f}",
        f"mAP50: {metrics.box.map50:.5f}",
        f"mAP50-95: {metrics.box.map:.5f}",
        "",
        "Per-class metrics",
    ]

    for row in rows:
        lines.extend(
            [
                f"Class: {row['class_name']} (ID {row['class_id']})",
                f"  Images: {row['images']}",
                f"  Instances: {row['instances']}",
                f"  Precision: {row['precision']:.5f}",
                f"  Recall: {row['recall']:.5f}",
                f"  F1: {row['f1']:.5f}",
                f"  mAP50: {row['mAP50']:.5f}",
                f"  mAP50-95: {row['mAP50-95']:.5f}",
                "",
            ]
        )

    TEXT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    validate_inputs()

    print("Evaluating the latest model on the held-out TEST split...")
    print(f"Device: {DEVICE}")

    model = YOLO(str(MODEL_PATH))
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        plots=True,
        project=str(OUTPUT_PROJECT),
        name=RUN_NAME,
        exist_ok=True,
    )

    rows = collect_class_metrics(metrics)
    if not rows:
        raise RuntimeError("Evaluation completed, but no per-class metrics were returned.")

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    save_csv_report(rows)
    save_text_report(metrics, rows)

    print("\nOverall test metrics")
    print(f"Precision: {metrics.box.mp:.5f}")
    print(f"Recall:    {metrics.box.mr:.5f}")
    print(f"mAP50:     {metrics.box.map50:.5f}")
    print(f"mAP50-95:  {metrics.box.map:.5f}")

    print("\nPer-class test metrics")
    for row in rows:
        print(
            f"{row['class_name']}: "
            f"P={row['precision']:.5f}, "
            f"R={row['recall']:.5f}, "
            f"F1={row['f1']:.5f}, "
            f"mAP50={row['mAP50']:.5f}, "
            f"mAP50-95={row['mAP50-95']:.5f}"
        )

    print(f"\nEvaluation artifacts: {REPORT_DIRECTORY}")
    print(f"CSV report: {CSV_REPORT_PATH}")
    print(f"Text report: {TEXT_REPORT_PATH}")


if __name__ == "__main__":
    main()
