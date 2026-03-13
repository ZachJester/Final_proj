# Thermal-RGB Fusion for YOLO Object Detection (MAPD Dataset)

This project demonstrates an approach to improve object detection—specifically for the `person` class—by fusing standard RGB camera frames with Thermal (infrared) imagery. The fusion leverages the strengths of both modalities, making detection robust to varying lighting conditions, occlusions, and backgrounds.

## How the Training Works

The training process is based on **Ultralytics YOLO** and utilizes a **4-channel dataset** (RGB + Thermal). Rather than modifying the YOLO architecture to natively accept 4 channels, the training pipeline uses an OpenCV-based preprocessing strategy to pre-fuse the multi-modal data into standard 3-channel (RGB) images that YOLO can readily digest.

### The Fusion Preprocessing Algorithm
The core of this approach lies in the preprocessing phase applied to every pair of RGB and Thermal images before they are fed into the model.

1. **Reading Data:**
   - The RGB image is read as a 3-channel color image: `(H, W, 3)`.
   - The corresponding Thermal image is read as a 1-channel grayscale image: `(H, W)`.

2. **Spatial Alignment (Resizing):**
   - The Thermal array is resized to match the exact spatial dimensions `(W, H)` of the RGB physical frame to ensure bounding box alignment between modalities.

3. **Normalization:**
   - To maximize contrast and ensure the thermal footprint spans the maximum pixel range, the Thermal image undergoes min-max normalization (`cv2.NORM_MINMAX`), stretching intensities to `0-255`.

4. **Channel Expansion:**
   - The normalized 1-channel Thermal image is duplicated across 3 channels, transforming it into a pseudo-RGB structure: `(H, W, 3)`.

5. **Weighted Additive Fusion:**
   - The dataset images are ultimately generated using a linear combination via `cv2.addWeighted`. 
   - **Weighting:** The RGB image holds a weight of `0.7` (`alpha`), and the expanded Thermal image holds `0.3` (`beta`), with a scalar offset of `0` (`gamma`).
   - `Fused = (0.7 * RGB) + (0.3 * Thermal_3CH)`
   - *Why this ratio?* This weighting retains the detailed structural and contextual textures provided by RGB while injecting enough thermal intensity (bright spots where body heat is detected) to highlight humans.

### Labels and Architecture
The original bounding box coordinates (mapped around the humans in the frames) from the dataset are applied directly to the newly outputted fused frames. The resulting dataset pipeline trains a standard YOLO model configured with a single class (`nc: 1`, names: `['person']`).

The training outputs the final PyTorch weights `best.pt`.

## How Inference Works

During inference, it is crucial that the model receives data identically processed to the training phase. 

The custom `inference.py` script included in this repository enforces this:
1. It accepts an arbitrary RGB image and its corresponding Thermal image.
2. It dynamically runs the **Fusion Preprocessing Algorithm** (Normalizing, 3-channel Thermal expanding, and `0.7/0.3` weighted fusion).
3. The YOLO model is loaded (`best.pt`) and performs inference on the fused image.
4. The output is a single image containing the fused visualization overlaid with the YOLO bounding boxes highlighting detected persons.

---
### See `setup.md` for a beginner-friendly guide to installing and running this project.

## Video Upload Inference API (People Count Stats)

This repo now includes a video inference pipeline that lets a user upload a video and get person-count stats.

### Files
- `video_inference.py`: frame-by-frame person counting with YOLO.
- `api.py`: FastAPI server with upload endpoint.
- `streamlit_app.py`: Streamlit UI for upload, inference settings, and visual outputs.

### Run the Streamlit UI
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

`streamlit_app.py` now has 2 modes:
- `People Video (YOLO .pt)`: current video + thermal workflow.
- `Plant Disease Images (.h5)`: upload a Keras `.h5` model and images for classification.

For `.h5` mode, install TensorFlow if not already available:
```bash
pip install tensorflow
```

### Run the API
```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints
- `GET /health`
- `POST /infer/video` (multipart form)

### `POST /infer/video` Form Fields
- `video` (file, required): input video file
- `thermal_video` (file, optional): synchronized thermal video (falls back to RGB-only if omitted)
- `frame_stride` (int, optional, default `5`): process every Nth frame
- `confidence` (float, optional, default `0.45`): YOLO confidence threshold
- `enable_tracking` (bool, optional, default `true`): enable ByteTrack IDs and unique-person counts
- `output_mode` (string, optional, default `stats`): one of `stats`, `frames`, `video`, `all`
- `include_timeline` (bool, optional, default `true`): include per-sampled-frame counts
- `save_annotated` (bool, optional, default `false`): save annotated output video

### Example Request
```bash
curl -X POST "http://localhost:8000/infer/video" \
  -F "video=@sample.mp4" \
  -F "thermal_video=@sample_thermal.mp4" \
  -F "frame_stride=5" \
  -F "confidence=0.45" \
  -F "enable_tracking=true" \
  -F "output_mode=frames" \
  -F "include_timeline=true" \
  -F "save_annotated=true"
```

If `output_mode=frames` or `all`, response includes `annotated_frame_urls`.
In `frames`/`all` mode, frame exports are generated for every frame with bounding boxes.
If `output_mode=video` or `all` (or `save_annotated=true`), response includes `annotated_video_url` served under `/outputs/...`.
