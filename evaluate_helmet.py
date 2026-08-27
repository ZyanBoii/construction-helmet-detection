"""Evaluate the trained helmet model on the held-out test split.

This script does not train the model. It measures generalization on images
that were not used during training or validation, then saves visual predictions.
"""

from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "construction_helmet_yolo" / "data.yaml"
MODEL_PATH = PROJECT_ROOT / "runs" / "construction_helmet_yolo26nExtra" / "weights" / "best.pt"
TEST_IMAGES = PROJECT_ROOT / "construction_helmet_yolo" / "images" / "test"
OUTPUT_PROJECT = PROJECT_ROOT / "runs" / "evaluation"


# The test set is used only for final measurement.
CONFIDENCE = 0.30
IMAGE_SIZE = 416
DEVICE = 0


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")
    if not TEST_IMAGES.is_dir():
        raise FileNotFoundError(f"Test images not found: {TEST_IMAGES}")

    model = YOLO(str(MODEL_PATH))

    print("Evaluating on the held-out TEST split...")
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=8,
        device=DEVICE,
        workers=0,
        plots=True,
        project=str(OUTPUT_PROJECT),
        name="test_metricsExtra",
        exist_ok=True,
    )

    print("\nTest metrics")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Per-class mAP50-95: {metrics.box.maps}")

    print("\nSaving test prediction images...")
    model.predict(
        source=str(TEST_IMAGES),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,
        save=True,
        save_conf=True,
        project=str(OUTPUT_PROJECT),
        name="test_predictions",
        exist_ok=True,
    )

    print(f"Metrics directory: {OUTPUT_PROJECT / 'test_metrics'}")
    print(f"Prediction directory: {OUTPUT_PROJECT / 'test_predictions'}")


if __name__ == "__main__":
    main()
