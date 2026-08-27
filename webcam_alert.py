"""Live construction helmet detection and no-helmet alert."""

from pathlib import Path
import time
import winsound

import cv2
from ultralytics import YOLO


# This file is inside the project root, so use a project-relative model path.
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "construction_helmet_yolo26nExtra"
    / "weights"
    / "best.pt"
)

CAMERA_INDEX = 0
IMAGE_SIZE = 416
INFERENCE_CONFIDENCE = 0.25
NO_HELMET_CONFIDENCE = 0.35
CONFIRMATION_SECONDS = 0.75
ALERT_COOLDOWN_SECONDS = 10.0
DEVICE = 0


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError(
            "Webcam could not be opened. Try CAMERA_INDEX = 1 or close other camera apps."
        )

    no_helmet_since = None
    last_alert_time = -float("inf")

    print("Webcam started. Press Q in the video window to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a webcam frame.")
                break

            result = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=INFERENCE_CONFIDENCE,
                device=DEVICE,
                verbose=False,
            )[0]

            now = time.monotonic()
            no_helmet_found = any(
                result.names[int(box.cls[0])] == "no_helmet"
                and float(box.conf[0]) >= NO_HELMET_CONFIDENCE
                for box in result.boxes
            )

            if no_helmet_found:
                if no_helmet_since is None:
                    no_helmet_since = now

                visible_for = now - no_helmet_since
                cooldown_finished = (
                    now - last_alert_time >= ALERT_COOLDOWN_SECONDS
                )

                if visible_for >= CONFIRMATION_SECONDS and cooldown_finished:
                    print("ALERT: No helmet detected")
                    winsound.Beep(1200, 500)
                    last_alert_time = now
            else:
                no_helmet_since = None

            display = result.plot()

            if no_helmet_found:
                cv2.putText(
                    display,
                    "WARNING: NO HELMET",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

            cv2.imshow("Construction Helmet Alert", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam stopped.")


if __name__ == "__main__":
    main()
