import base64
import importlib
import json
import uuid
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from video_inference import VideoPeopleCounter


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "streamlit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_MODEL_PATH = BASE_DIR / "best.pt"
PLANT_MODEL_PATH = BASE_DIR / "plant_disease_model.h5"


@st.cache_resource
def load_counter(model_path: str) -> VideoPeopleCounter:
    return VideoPeopleCounter(model_path)


@st.cache_resource
def load_keras_model(model_path: str):
    # Lazy import so video mode still works without TensorFlow installed.
    keras_models = importlib.import_module("tensorflow.keras.models")
    return keras_models.load_model(model_path, compile=False)


def save_upload(uploaded_file, out_path: Path) -> None:
    out_path.write_bytes(uploaded_file.getbuffer())


def render_image_inline(image_path: Path, caption: str) -> None:
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    st.markdown(
        (
            f'<figure style="margin: 0;">'
            f'<img src="data:image/jpeg;base64,{encoded}" style="width: 100%; height: auto;" />'
            f'<figcaption style="color: #888; font-size: 0.9rem;">{caption}</figcaption>'
            f"</figure>"
        ),
        unsafe_allow_html=True,
    )


def parse_class_names(raw_text: str) -> list[str]:
    if not raw_text.strip():
        return []
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def preprocess_for_keras(image: Image.Image, target_h: int, target_w: int, normalize: bool) -> np.ndarray:
    rgb = image.convert("RGB").resize((target_w, target_h))
    arr = np.asarray(rgb, dtype=np.float32)
    if normalize:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def run_video_mode() -> None:
    st.title("Video People Tracking Inference")
    st.caption("Upload RGB video with optional thermal video, then run fused person detection + tracking.")

    if not VIDEO_MODEL_PATH.exists():
        st.error(f"Model file not found: {VIDEO_MODEL_PATH}")
        return

    counter = load_counter(str(VIDEO_MODEL_PATH))
    if "last_video_result" not in st.session_state:
        st.session_state.last_video_result = None

    with st.sidebar:
        st.header("Video Settings")
        frame_stride = st.slider("Frame stride", min_value=1, max_value=30, value=5, step=1)
        confidence = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.45, step=0.01)
        enable_tracking = st.checkbox("Enable tracking", value=True)
        output_mode = st.selectbox("Output mode", options=["stats", "frames", "video", "all"], index=0)
        include_timeline = st.checkbox("Include timeline in stats", value=True)

    col1, col2 = st.columns(2)
    with col1:
        rgb_video = st.file_uploader("RGB video", type=["mp4", "mov", "avi", "mkv"], key="rgb_video")
    with col2:
        thermal_video = st.file_uploader(
            "Thermal video (optional)", type=["mp4", "mov", "avi", "mkv"], key="thermal_video"
        )

    run_clicked = st.button("Run Video Inference", type="primary", disabled=rgb_video is None)
    if run_clicked:
        if rgb_video is None:
            st.warning("Please upload an RGB video.")
            return

        job_id = uuid.uuid4().hex[:12]
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = job_dir / rgb_video.name
        save_upload(rgb_video, rgb_path)

        thermal_path = None
        if thermal_video is not None:
            thermal_path = job_dir / thermal_video.name
            save_upload(thermal_video, thermal_path)

        want_frames = output_mode in {"frames", "all"}
        want_video = output_mode in {"video", "all"}
        annotated_video_path = job_dir / "annotated_output.mp4"
        frames_dir = job_dir / "frames"

        with st.spinner("Running video model..."):
            stats = counter.analyze(
                video_path=str(rgb_path),
                thermal_video_path=str(thermal_path) if thermal_path else None,
                frame_stride=frame_stride,
                confidence=confidence,
                enable_tracking=enable_tracking,
                save_annotated=want_video,
                annotated_output_path=str(annotated_video_path) if want_video else None,
                save_annotated_frames=want_frames,
                frames_output_dir=str(frames_dir) if want_frames else None,
                annotate_all_frames=want_frames,
            )

        if not include_timeline:
            stats.pop("timeline", None)

        st.session_state.last_video_result = {
            "job_id": job_id,
            "stats": stats,
            "want_video": want_video,
            "want_frames": want_frames,
            "annotated_video_path": str(annotated_video_path),
        }

    result = st.session_state.last_video_result
    if result is None:
        st.info("Upload a video and click 'Run Video Inference' to see results.")
        return

    job_id = result["job_id"]
    stats = result["stats"]
    want_video = result["want_video"]
    want_frames = result["want_frames"]
    annotated_video_path = Path(result["annotated_video_path"])

    st.subheader("Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unique People", stats.get("unique_people_tracked", 0))
    m2.metric("Total Detections", stats.get("total_people_detections", 0))
    m3.metric("Max In Frame", stats.get("max_people_in_frame", 0))
    m4.metric("Avg Confidence", stats.get("average_person_confidence", 0.0))

    st.subheader("Stats JSON")
    st.json(stats)
    st.download_button(
        "Download stats.json",
        data=json.dumps(stats, indent=2),
        file_name=f"{job_id}_stats.json",
        mime="application/json",
    )

    if want_video and annotated_video_path.exists():
        st.subheader("Annotated Video")
        st.video(str(annotated_video_path))

    if want_frames:
        frame_paths = stats.get("annotated_frame_paths", [])
        if frame_paths:
            st.subheader("Annotated Frames")
            selected = st.slider(
                "Frame index",
                min_value=0,
                max_value=len(frame_paths) - 1,
                value=0,
                key=f"frame_slider_{job_id}",
            )
            selected_path = Path(frame_paths[selected])
            if selected_path.exists():
                render_image_inline(selected_path, selected_path.name)
            else:
                st.warning(f"Frame file not found: {selected_path}")


def run_plant_mode() -> None:
    st.title("Plant Disease Detection (.h5)")
    st.caption("Uses a relative model path by default: ./plant_disease_model.h5")

    with st.sidebar:
        st.header("Plant Settings")
        normalize = st.checkbox("Normalize image to [0,1]", value=True)

    model_file = st.file_uploader(
        "Plant model override (.h5, optional)", type=["h5"], key="plant_model"
    )
    label_file = st.file_uploader("Class names (.txt, optional)", type=["txt"], key="plant_labels")
    images = st.file_uploader(
        "Plant images", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True, key="plant_images"
    )

    class_text_default = ""
    if label_file is not None:
        class_text_default = label_file.getvalue().decode("utf-8", errors="ignore")
    class_text = st.text_area("Class names (one per line, optional)", value=class_text_default, height=120)

    model_available = PLANT_MODEL_PATH.exists() or model_file is not None
    disabled = (not model_available) or (not images)
    run_clicked = st.button("Run Plant Inference", type="primary", disabled=disabled)
    if not run_clicked:
        if not model_available:
            st.warning(f"Missing default model at: {PLANT_MODEL_PATH}")
        return

    if not images:
        st.warning("Please upload at least one image.")
        return

    if model_file is not None:
        model_job_id = uuid.uuid4().hex[:12]
        model_dir = OUTPUT_DIR / "plant_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{model_job_id}.h5"
        save_upload(model_file, model_path)
    else:
        model_path = PLANT_MODEL_PATH

    try:
        with st.spinner("Loading .h5 model..."):
            keras_model = load_keras_model(str(model_path))
    except Exception as exc:
        st.error(f"Failed to load .h5 model: {exc}")
        st.info("Install TensorFlow in your environment (e.g., `pip install tensorflow` or `tensorflow-cpu`).")
        return

    input_shape = keras_model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if len(input_shape) != 4:
        st.error(f"Expected image model input shape [batch, h, w, c], got: {input_shape}")
        return

    target_h = int(input_shape[1])
    target_w = int(input_shape[2])
    class_names = parse_class_names(class_text)

    rows = []
    preview_cols = st.columns(3)

    for idx, img_file in enumerate(images):
        pil_img = Image.open(img_file)
        batch = preprocess_for_keras(pil_img, target_h, target_w, normalize)

        pred = keras_model.predict(batch, verbose=0)
        probs = np.asarray(pred).reshape(-1)
        pred_idx = int(np.argmax(probs))
        pred_conf = float(probs[pred_idx])
        pred_label = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"

        rows.append(
            {
                "image": img_file.name,
                "predicted_index": pred_idx,
                "predicted_label": pred_label,
                "confidence": round(pred_conf, 4),
            }
        )

        with preview_cols[idx % 3]:
            st.image(pil_img, caption=f"{img_file.name} -> {pred_label} ({pred_conf:.3f})", width="stretch")

    st.subheader("Predictions")
    st.dataframe(rows, width="stretch")

    st.download_button(
        "Download predictions.json",
        data=json.dumps(rows, indent=2),
        file_name="plant_predictions.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(page_title="Inference Dashboard", layout="wide")

    mode = st.sidebar.radio(
        "Mode",
        options=["People Video (YOLO .pt)", "Plant Disease Images (.h5)"],
        index=0,
    )

    if mode == "People Video (YOLO .pt)":
        run_video_mode()
    else:
        run_plant_mode()


if __name__ == "__main__":
    main()




