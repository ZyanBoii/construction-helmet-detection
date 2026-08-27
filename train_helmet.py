"""Train the construction safety-helmet detector with Ultralytics YOLO.

Start with SMOKE_TEST = True. This runs only one epoch so that we can verify
the dataset, labels, GPU, and training pipeline before spending hours training.
"""

from pathlib import Path

import torch
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Experiment settings: change values here instead of rewriting the code below.
# ---------------------------------------------------------------------------

SMOKE_TEST = False

MODEL_NAME = "yolo26n.pt"
IMAGE_SIZE = 416
BATCH_SIZE = 8
DEVICE = 0  # 0 = first NVIDIA GPU; use "cpu" only when CUDA is unavailable.
WORKERS = 0  # A safe starting value for Windows.
RANDOM_SEED = 42

# Full-training settings. They are used when SMOKE_TEST is False.
FULL_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 20


# Resolve paths from this file, so the script works regardless of the terminal's
# current directory.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "construction_helmet_yolo" / "data.yaml"
RUNS_DIRECTORY = PROJECT_ROOT / "runs"


def validate_environment() -> None:
    """Fail early with a clear message when a required input is unavailable."""
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset configuration was not found: {DATA_YAML}")

    if DEVICE == 0 and not torch.cuda.is_available():
        raise RuntimeError(
            "DEVICE=0 requests an NVIDIA GPU, but PyTorch cannot access CUDA. "
            "Check the PyTorch/CUDA installation or temporarily set DEVICE='cpu'."
        )


def main() -> None:
    validate_environment()

    epochs = 1 if SMOKE_TEST else FULL_EPOCHS
    run_name = "smoke_test" if SMOKE_TEST else "construction_helmet_yolo26nExtra"

    print(f"Mode: {'smoke test' if SMOKE_TEST else 'full training'}")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Epochs: {epochs}")
    print(f"CUDA GPU: {torch.cuda.get_device_name(DEVICE)}")

    # .pt means that we begin with pretrained weights (transfer learning).
    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        patience=EARLY_STOPPING_PATIENCE,
        project=str(RUNS_DIRECTORY),
        name=run_name,
        seed=RANDOM_SEED,
        plots=True,
        exist_ok=False,
    )

    # Ultralytics appends a suffix such as "-2" when a run name already exists.
    # Read the actual save directory from the trainer instead of guessing it.
    output_directory = Path(model.trainer.save_dir)
    print("\nTraining completed successfully.")
    print(f"Review the results in: {output_directory}")
    print(f"Best model: {output_directory / 'weights' / 'best.pt'}")


# This guard is important when Ultralytics/PyTorch runs on Windows.
if __name__ == "__main__":
    main()
