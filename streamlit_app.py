
   # -m streamlit run streamlit_app.py
from pathlib import Path
import os
import threading

import av
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Keep Ultralytics settings inside this project on Windows.
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parent / "Ultralytics"),
)

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "construction_helmet_yolo26nExtra" #construction_helmet_yolo26n
    / "weights"
    / "best.pt"
)
INFERENCE_DEVICE = 0 if torch.cuda.is_available() else "cpu"
MODEL_INFERENCE_LOCK = threading.Lock()


@st.cache_resource # for : not run from start when user input (for reduce memory)
def load_model() -> YOLO:
    """Load the model once instead of loading it on every Streamlit rerun."""
    return YOLO(str(MODEL_PATH))


def show_live_camera(model: YOLO, confidence: float) -> None:
    """Run browser-camera inference and return annotated WebRTC frames."""
    st.subheader("Live camera detection (beta)")
    st.caption(
        "Click START and allow camera access. Cloud inference may have lower "
        "frame rate because it normally runs on CPU."
    )

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        frame_bgr = frame.to_ndarray(format="bgr24")

        # The cached model can be shared by callbacks, so protect prediction.
        with MODEL_INFERENCE_LOCK:
            result = model.predict(
                source=frame_bgr,
                imgsz=416,
                conf=confidence,
                device=INFERENCE_DEVICE,
                verbose=False,
            )[0]

        annotated = result.plot()
        helmet_count = 0
        no_helmet_count = 0

        for box in result.boxes:
            class_name = result.names[int(box.cls[0])]
            if class_name == "helmet":
                helmet_count += 1
            elif class_name == "no_helmet":
                no_helmet_count += 1

        if no_helmet_count > 0:
            status_text = "WARNING: NO HELMET DETECTED"
            status_color = (0, 0, 255)
        elif helmet_count > 0:
            status_text = "HELMET DETECTED"
            status_color = (0, 170, 0)
        else:
            status_text = "NO HEAD DETECTED"
            status_color = (0, 190, 255)

        cv2.rectangle(annotated, (8, 8), (510, 82), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            status_text,
            (18, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            status_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Helmet: {helmet_count}   No helmet: {no_helmet_count}",
            (18, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="construction-helmet-live-camera",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
        async_processing=True,
    )

    st.warning(
        "Live mode is an educational beta. If START cannot connect on a "
                "restricted network, use the camera snapshot or image upload mode. "
                "  \n**Thanks a million for using my Demo.**  \n"
                "Zyanobii (Nay Lin)"
    )

def main() -> None:
    st.set_page_config(
        page_title="Construction Helmet Alert",
        page_icon="🦺",
        layout="wide",
        #layout="centered",
    )

    st.title("Construction Safety Helmet Detect and Alert")
    st.write(
        "Upload an image or take a camera snapshot. "
        "The model checks for helmet and no-helmet detections. ^_^"
    )
    st.caption(
        "University demonstration only. Do not use this application as the "
        "sole construction-safety control."
    )

    if not MODEL_PATH.is_file():
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()

    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Increase this value if false alerts are too frequent.",
    )

    source_type = st.radio(
        "Choose an input",
        ["Upload image", "Use camera snapshot", "Live camera (beta)"],
        horizontal=True,
    )

    model = load_model()

    if source_type == "Live camera (beta)":
        show_live_camera(model, confidence)
        return

    uploaded_file = None
    if source_type == "Upload image":
        uploaded_file = st.file_uploader(
            "Choose a JPG or PNG image",
            type=["jpg", "jpeg", "png"],
        )
    else:
        uploaded_file = st.camera_input("Take a picture")

    if uploaded_file is None:
        st.info("Choose an image or take a camera snapshot to begin.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.asarray(image)
    # Ultralytics treats NumPy image sources as OpenCV-style BGR.
    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])

    with st.spinner("Detecting..."):
        result = model.predict(
            source=image_bgr,
            imgsz=416,
            conf=confidence,
            device=INFERENCE_DEVICE,
            verbose=False,
        )[0]

    # Request an RGB PIL image directly to avoid another BGR/RGB mismatch.
    annotated_image = result.plot(pil=True)
    st.image(annotated_image, caption="Detection result", use_container_width=True)

    no_helmet_count = 0
    helmet_count = 0
    for box in result.boxes:
        class_name = result.names[int(box.cls[0])]
        if class_name == "no_helmet":
            no_helmet_count += 1
        elif class_name == "helmet":
            helmet_count += 1

    st.metric("Helmet", helmet_count)
    st.metric("No helmet", no_helmet_count)

    if no_helmet_count > 0:
        st.error("⚠️ ALERT: No helmet detected")
    elif helmet_count > 0:
        st.success("✅ No no-helmet detection")
    else:
        st.warning("nothing detect , try another")

if __name__ == "__main__":
    main()
