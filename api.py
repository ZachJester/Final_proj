import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from video_inference import VideoPeopleCounter


app = FastAPI(title="Video People Counting API", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = BASE_DIR / "best.pt"
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found: {MODEL_PATH}")

COUNTER = VideoPeopleCounter(str(MODEL_PATH))

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/health")
def health():
    return {"status": "ok", "model": str(MODEL_PATH.name)}


@app.post("/infer/video")
async def infer_video(
    video: UploadFile = File(...),
    thermal_video: UploadFile | None = File(None),
    frame_stride: int = Form(5),
    confidence: float = Form(0.45),
    enable_tracking: bool = Form(True),
    output_mode: str = Form("stats"),
    include_timeline: bool = Form(True),
    save_annotated: bool = Form(False),
):
    if frame_stride < 1:
        raise HTTPException(status_code=400, detail="frame_stride must be >= 1")
    if confidence < 0 or confidence > 1:
        raise HTTPException(status_code=400, detail="confidence must be between 0 and 1")
    output_mode = output_mode.lower().strip()
    allowed_output_modes = {"stats", "frames", "video", "all"}
    if output_mode not in allowed_output_modes:
        raise HTTPException(
            status_code=400,
            detail="output_mode must be one of: stats, frames, video, all",
        )

    suffix = Path(video.filename or "uploaded.mp4").suffix or ".mp4"
    thermal_suffix = ".mp4"
    job_id = uuid.uuid4().hex[:12]
    upload_path = UPLOAD_DIR / f"{job_id}{suffix}"
    thermal_upload_path = None
    annotated_path = OUTPUT_DIR / f"{job_id}_annotated.mp4"
    frames_dir = OUTPUT_DIR / f"{job_id}_frames"
    want_frames = output_mode in {"frames", "all"}
    want_video = save_annotated or output_mode in {"video", "all"}

    with upload_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    if thermal_video is not None:
        thermal_suffix = Path(thermal_video.filename or "thermal.mp4").suffix or ".mp4"
        thermal_upload_path = UPLOAD_DIR / f"{job_id}_thermal{thermal_suffix}"
        with thermal_upload_path.open("wb") as f:
            shutil.copyfileobj(thermal_video.file, f)

    try:
        stats = COUNTER.analyze(
            video_path=str(upload_path),
            thermal_video_path=str(thermal_upload_path) if thermal_upload_path else None,
            frame_stride=frame_stride,
            confidence=confidence,
            enable_tracking=enable_tracking,
            save_annotated=want_video,
            annotated_output_path=str(annotated_path) if want_video else None,
            save_annotated_frames=want_frames,
            frames_output_dir=str(frames_dir) if want_frames else None,
            annotate_all_frames=want_frames,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        video.file.close()
        if thermal_video is not None:
            thermal_video.file.close()

    if not include_timeline:
        stats.pop("timeline", None)

    response = {
        "job_id": job_id,
        "filename": video.filename,
        "thermal_filename": thermal_video.filename if thermal_video is not None else None,
        "output_mode": output_mode,
        "stats": stats,
    }

    if want_video:
        response["annotated_video_url"] = f"/outputs/{annotated_path.name}"
    if want_frames:
        frame_paths = stats.get("annotated_frame_paths", [])
        response["annotated_frame_urls"] = [
            f"/outputs/{frames_dir.name}/{Path(frame_path).name}" for frame_path in frame_paths
        ]

    return response
