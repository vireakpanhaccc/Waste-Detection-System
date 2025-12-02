"""
Inference Script for Waste Detection
Run predictions on new images or video
"""

from ultralytics import YOLO
import torch
import argparse
from pathlib import Path
import cv2
import json

def predict_image(model_path, image_path, conf_threshold=0.25, save_dir='predictions'):
    """
    Run prediction on a single image
    
    Args:
        model_path: Path to trained model
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections
        save_dir: Directory to save results
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = YOLO(model_path)
    
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        project=save_dir,
        name='results',
        device=device
    )
    
    # Print detection results
    for r in results:
        print(f"\nDetections in {image_path}:")
        if len(r.boxes) == 0:
            print("  No objects detected")
        else:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                print(f"  - {class_name}: {confidence:.2f}")
    
    return results

def predict_folder(model_path, folder_path, conf_threshold=0.25, save_dir='predictions'):
    """
    Run predictions on all images in a folder
    
    Args:
        model_path: Path to trained model
        folder_path: Path to folder containing images
        conf_threshold: Confidence threshold
        save_dir: Directory to save results
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = YOLO(model_path)
    
    folder = Path(folder_path)
    image_files = list(folder.glob('*.jpg')) + list(folder.glob('*.png')) + \
                  list(folder.glob('*.jpeg')) + list(folder.glob('*.JPG'))
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images")
    
    results = model.predict(
        source=str(folder),
        conf=conf_threshold,
        save=True,
        save_txt=True,
        save_conf=True,
        project=save_dir,
        name='results',
        device=device
    )
    
    # Summary statistics
    total_detections = sum(len(r.boxes) for r in results)
    print(f"\n✓ Processed {len(image_files)} images")
    print(f"✓ Total detections: {total_detections}")
    print(f"✓ Results saved to: {save_dir}/results")
    
    return results

def predict_video(model_path, video_path, conf_threshold=0.25, save_dir='predictions'):
    """
    Run predictions on video
    
    Args:
        model_path: Path to trained model
        video_path: Path to input video
        conf_threshold: Confidence threshold
        save_dir: Directory to save results
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = YOLO(model_path)
    
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        save=True,
        project=save_dir,
        name='video_results',
        device=device,
        stream=True  # Use streaming for videos
    )
    
    frame_count = 0
    for r in results:
        frame_count += 1
        if frame_count % 30 == 0:  # Print every 30 frames
            print(f"Processed {frame_count} frames...")
    
    print(f"\n✓ Processed {frame_count} frames")
    print(f"✓ Results saved to: {save_dir}/video_results")

def predict_webcam(model_path, conf_threshold=0.25):
    """
    Run real-time predictions from webcam
    
    Args:
        model_path: Path to trained model
        conf_threshold: Confidence threshold
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("Starting webcam... Press 'q' to quit")
    
    model = YOLO(model_path)
    
    # Run inference on webcam
    results = model.predict(
        source=0,  # 0 for default webcam
        conf=conf_threshold,
        show=True,  # Display results
        device=device,
        stream=True
    )
    
    for r in results:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run waste detection predictions')
    parser.add_argument('--model', type=str, default='runs/detect/waste_detection_v1/weights/best.pt',
                       help='Path to trained model weights')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image, folder, video, or "webcam"')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--save-dir', type=str, default='predictions',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    
    if args.source.lower() == 'webcam':
        predict_webcam(args.model, args.conf)
    elif source_path.is_file():
        # Check if video or image
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        if source_path.suffix.lower() in video_extensions:
            predict_video(args.model, args.source, args.conf, args.save_dir)
        else:
            predict_image(args.model, args.source, args.conf, args.save_dir)
    elif source_path.is_dir():
        predict_folder(args.model, args.source, args.conf, args.save_dir)
    else:
        print(f"Error: Invalid source path: {args.source}")
