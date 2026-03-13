# Setup and Run Guide

This project now supports four inference paths:
- Image fusion inference (`inference.py`)
- Video people counting/tracking CLI (`video_inference.py`)
- Video upload API (`api.py`)
- Streamlit UI with two modes:
  - People video (`.pt` YOLO)
  - Plant disease image inference (`.h5` Keras)

## 1. Create and Activate a Virtual Environment

```bash
python -m venv .venv
```

Windows (PowerShell):
```powershell
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If you plan to use plant `.h5` mode in Streamlit and TensorFlow is missing:

```bash
pip install tensorflow
```

## 3. Run Image Fusion Inference (RGB + Thermal Image Pair)

```bash
python inference.py --rgb sample_rgb.jpg --thermal sample_thermal.jpg --out inference_result.jpg
```

Output: `inference_result.jpg`

## 4. Run Video CLI Inference (People Counting + Tracking)

RGB-only:
```bash
python video_inference.py --video input.mp4
```

RGB + Thermal video:
```bash
python video_inference.py --video rgb.mp4 --thermal-video thermal.mp4
```

Useful options:
- `--conf 0.45`
- `--stride 5`
- `--save-annotated --annotated-out annotated_output.mp4`
- `--no-track` (disable unique-person tracking)

## 5. Run FastAPI Video Service

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Health check:
```bash
curl http://localhost:8000/health
```

Video inference request:
```bash
curl -X POST "http://localhost:8000/infer/video" \
  -F "video=@sample.mp4" \
  -F "frame_stride=5" \
  -F "confidence=0.45" \
  -F "enable_tracking=true" \
  -F "output_mode=frames"
```

`output_mode` supports: `stats`, `frames`, `video`, `all`.

## 6. Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

### Streamlit Modes

1. `People Video (YOLO .pt)`
- Uses `best.pt`
- Supports optional thermal video
- Supports tracking and output modes (`stats/frames/video/all`)

2. `Plant Disease Images (.h5)`
- Uses `./plant_disease_model.h5` by default (relative path)
- Optional uploaded `.h5` can override default model
- Accepts multiple input images
- Optional class names from `.txt` or text box

## 7. Common Notes

- Large media/model files are ignored by default (`*.mp4`, `*.h5`) in `.gitignore`.
- If dependency resolver errors occur, ensure your venv is active and retry:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
