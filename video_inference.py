import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics.models.yolo.model import YOLO


class VideoPeopleCounter:
    def __init__(self, model_path: str = "best.pt"):
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.person_class_ids = self._person_ids(self.names)

    @staticmethod
    def _person_ids(names):
        person_ids = set()
        if isinstance(names, dict):
            for idx, name in names.items():
                if str(name).lower() == "person":
                    person_ids.add(int(idx))
        elif isinstance(names, (list, tuple)):
            for idx, name in enumerate(names):
                if str(name).lower() == "person":
                    person_ids.add(idx)

        # Fallback for single-class person-only models
        if not person_ids:
            if isinstance(names, dict) and len(names) == 1:
                person_ids.add(int(next(iter(names.keys()))))
            elif isinstance(names, (list, tuple)) and len(names) == 1:
                person_ids.add(0)

        return person_ids

    def analyze(
        self,
        video_path: str,
        thermal_video_path: str | None = None,
        frame_stride: int = 5,
        confidence: float = 0.45,
        enable_tracking: bool = True,
        save_annotated: bool = False,
        annotated_output_path: str | None = None,
        save_annotated_frames: bool = False,
        frames_output_dir: str | None = None,
        annotate_all_frames: bool = False,
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        thermal_cap = None
        use_thermal = False
        if thermal_video_path:
            thermal_cap = cv2.VideoCapture(thermal_video_path)
            if not thermal_cap.isOpened():
                cap.release()
                raise ValueError(f"Could not open thermal video file: {thermal_video_path}")
            use_thermal = True

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save_annotated:
            if not annotated_output_path:
                raise ValueError("annotated_output_path is required when save_annotated=True")
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            writer = cv2.VideoWriter(annotated_output_path, fourcc, fps or 25.0, (width, height))
        frame_output_path = None
        if save_annotated_frames:
            if not frames_output_dir:
                raise ValueError("frames_output_dir is required when save_annotated_frames=True")
            frame_output_path = Path(frames_output_dir)
            frame_output_path.mkdir(parents=True, exist_ok=True)

        started = time.time()
        frame_idx = 0
        processed_frames = 0
        total_people_detections = 0
        max_people = 0
        peak_frame = 0
        confidence_sum = 0.0
        confidence_count = 0
        timeline = []
        thermal_frames_used = 0
        unique_track_ids = set()
        tracked_detections = 0
        annotated_frame_paths = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            inference_frame = frame
            if thermal_cap is not None:
                thermal_ok, thermal_frame = thermal_cap.read()
                if thermal_ok:
                    # Convert to grayscale thermal channel, align shape, normalize and fuse.
                    thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
                    thermal_gray = cv2.resize(thermal_gray, (frame.shape[1], frame.shape[0]))
                    thermal_gray = cv2.normalize(
                        thermal_gray, thermal_gray, 0, 255, cv2.NORM_MINMAX
                    )
                    thermal_3ch = np.repeat(thermal_gray[:, :, None], 3, axis=2)
                    inference_frame = cv2.addWeighted(frame, 0.7, thermal_3ch, 0.3, 0)
                    thermal_frames_used += 1

            run_inference_for_stats = frame_stride <= 1 or (frame_idx % frame_stride == 0)
            run_inference = run_inference_for_stats or annotate_all_frames
            if run_inference:
                if enable_tracking:
                    results = self.model.track(
                        inference_frame,
                        conf=confidence,
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False,
                    )
                else:
                    results = self.model(inference_frame, conf=confidence, verbose=False)
                result = results[0]

                count = 0
                if result.boxes is not None and len(result.boxes) > 0:
                    classes = result.boxes.cls.tolist()
                    confs = result.boxes.conf.tolist()
                    if result.boxes.id is not None:
                        track_ids = result.boxes.id.tolist()
                    else:
                        track_ids = [None] * len(classes)

                    for cls_id, conf_score, track_id in zip(classes, confs, track_ids):
                        if int(cls_id) in self.person_class_ids:
                            count += 1
                            confidence_sum += float(conf_score)
                            confidence_count += 1
                            if track_id is not None:
                                unique_track_ids.add(int(track_id))
                                tracked_detections += 1

                if run_inference_for_stats:
                    processed_frames += 1
                    total_people_detections += count
                    if count > max_people:
                        max_people = count
                        peak_frame = frame_idx

                    timestamp = frame_idx / fps if fps else None
                    timeline.append(
                        {
                            "frame": frame_idx,
                            "timestamp_sec": round(timestamp, 3) if timestamp is not None else None,
                            "people_count": count,
                            "unique_people_so_far": len(unique_track_ids),
                        }
                    )

                annotated_frame = None
                if writer is not None:
                    annotated_frame = result.plot()
                    writer.write(annotated_frame)
                if frame_output_path is not None:
                    if annotated_frame is None:
                        annotated_frame = result.plot()
                    frame_file = frame_output_path / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_file), annotated_frame)
                    annotated_frame_paths.append(str(frame_file))
            elif writer is not None:
                writer.write(inference_frame)

            frame_idx += 1

        cap.release()
        if thermal_cap is not None:
            thermal_cap.release()
        if writer is not None:
            writer.release()

        elapsed = time.time() - started
        avg_people = (total_people_detections / processed_frames) if processed_frames else 0.0
        avg_conf = (confidence_sum / confidence_count) if confidence_count else 0.0

        return {
            "video_path": str(video_path),
            "thermal_video_path": str(thermal_video_path) if thermal_video_path else None,
            "fusion_mode": "rgb_thermal" if use_thermal else "rgb_only",
            "thermal_frames_used": thermal_frames_used,
            "fps": round(fps, 3),
            "resolution": {"width": width, "height": height},
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "frame_stride": frame_stride,
            "annotate_all_frames": annotate_all_frames,
            "duration_sec": round((total_frames / fps), 3) if fps else None,
            "total_people_detections": int(total_people_detections),
            "tracking_enabled": enable_tracking,
            "unique_people_tracked": len(unique_track_ids),
            "tracked_person_detections": int(tracked_detections),
            "max_people_in_frame": int(max_people),
            "average_people_per_processed_frame": round(avg_people, 3),
            "average_person_confidence": round(avg_conf, 3),
            "peak_frame": int(peak_frame),
            "peak_time_sec": round((peak_frame / fps), 3) if fps else None,
            "processing_time_sec": round(elapsed, 3),
            "timeline": timeline,
            "annotated_video_path": annotated_output_path if save_annotated else None,
            "annotated_frame_paths": annotated_frame_paths if save_annotated_frames else [],
        }


def main():
    parser = argparse.ArgumentParser(description="Video inference pipeline for person counting")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
        "--thermal-video",
        default=None,
        help="Optional path to thermal video; if omitted, RGB-only inference is used",
    )
    parser.add_argument("--model", default="best.pt", help="Path to YOLO model")
    parser.add_argument("--stride", type=int, default=5, help="Process every Nth frame")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable tracker and run plain detection only",
    )
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated output video")
    parser.add_argument("--annotated-out", default="annotated_output.mp4", help="Annotated output path")

    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    analyzer = VideoPeopleCounter(args.model)
    stats = analyzer.analyze(
        video_path=args.video,
        thermal_video_path=args.thermal_video,
        frame_stride=args.stride,
        confidence=args.conf,
        enable_tracking=not args.no_track,
        save_annotated=args.save_annotated,
        annotated_output_path=args.annotated_out if args.save_annotated else None,
    )

    print("Video stats:")
    for key, value in stats.items():
        if key == "timeline":
            print(f"{key}: {len(value)} sampled points")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
