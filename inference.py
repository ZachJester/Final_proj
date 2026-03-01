import cv2
import numpy as np
import argparse
from ultralytics.models.yolo.model import YOLO


def preprocess_fused(rgb_path, thermal_path):
    """
    Applies the same preprocessing used during training:
    1. Reads RGB and Grayscale Thermal images.
    2. Resizes Thermal to match RGB.
    3. Normalizes Thermal.
    4. Expands Thermal to 3 channels.
    5. Performs 0.7/0.3 weighted fusion.
    """
    # 1. Read images
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)  # (H, W, 3)
    thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)  # (H, W)

    if rgb is None:
        raise FileNotFoundError(f"Could not load RGB image: {rgb_path}")
    if thermal is None:
        raise FileNotFoundError(f"Could not load Thermal image: {thermal_path}")

    # 2. Resize thermal to match RGB if sizes differ
    thermal = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]))

    # 3. Normalize thermal to 0-255 using MINMAX
    thermal = cv2.normalize(thermal, thermal, 0, 255, cv2.NORM_MINMAX)

    # 4. Expand thermal to 3 channels to match RGB shape
    thermal_3ch = np.repeat(thermal[:, :, None], 3, axis=2)

    # 5. Weighted fusion (using the same alpha/beta constants from training)
    fused = cv2.addWeighted(rgb, 0.7, thermal_3ch, 0.3, 0)

    return fused


def infer(
    rgb_path, thermal_path, model_path="best.pt", output_path="inference_result.jpg"
):
    """
    Fuses the images and runs model inference using YOLO.
    """
    try:
        # Preprocess
        print("Preprocessing images...")
        print(f"  RGB: {rgb_path}")
        print(f"  Thermal: {thermal_path}")
        fused_img = preprocess_fused(rgb_path, thermal_path)

        # Load Model
        print(f"\nLoading model from {model_path}...")
        model = YOLO(model_path)

        # Run Inference
        print("Running inference...")
        results = model(fused_img)

        # Plot results on the fused image (with bounding boxes and labels)
        annotated_img = results[0].plot()

        if output_path is not None:
            cv2.imwrite(output_path, annotated_img)
            print(f"\nSuccess! Inference saved to: {output_path}")
        else:
            # Display if no output path provided
            cv2.imshow("Inference Result", annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"\nError during inference: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for fused RGB-Thermal YOLO model."
    )
    parser.add_argument("--rgb", type=str, required=True, help="Path to the RGB image")
    parser.add_argument(
        "--thermal", type=str, required=True, help="Path to the Thermal image"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best.pt",
        help="Path to the trained PyTorch model (.pt)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="inference_result.jpg",
        help="Output path for the annotated image",
    )

    args = parser.parse_args()

    # Run the process
    infer(args.rgb, args.thermal, args.model, args.out)
