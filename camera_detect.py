"""
Real-time Waste Detection using Webcam with OpenCV
This script opens your camera and detects waste objects in real-time
"""

import cv2
from ultralytics import YOLO
import torch
import os
import time

class WasteDetector:
    def __init__(self, model_path='best.pt', 
                 conf_threshold=0.5):
        """
        Initialize the waste detector
        
        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold for detections (0-1)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}\nPlease train the model first.")
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Check device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Get class names
        self.class_names = self.model.names
        
        # Simplify plastic type names for display
        self.display_names = {}
        plastic_types = {'HDPE', 'LDPE', 'PETE', 'PP', 'PS', 'PVC'}
        
        for idx, name in self.class_names.items():
            if name in plastic_types:
                self.display_names[idx] = 'Plastic'
            else:
                self.display_names[idx] = name.title()
        
        print(f"\nDetectable waste types:")
        for idx, name in self.class_names.items():
            display = self.display_names[idx]
            if name in plastic_types:
                print(f"  {idx}: {name} → {display}")
            else:
                print(f"  {idx}: {display}")
    
    def detect_from_camera(self, camera_id=0, window_name="Waste Detection"):
        """
        Run real-time detection from webcam
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
            window_name: Name of the display window
        """
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"\n{'='*70}")
        print("CAMERA STARTED - Real-time Waste Detection")
        print(f"{'='*70}")
        print("Controls:")
        print("  Press 'q' to quit")
        print("  Press 's' to save screenshot")
        print(f"{'='*70}\n")
        
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        screenshot_count = 0
        
        try:
            while True:
                frame_start = time.time()
                
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Cannot read frame from camera")
                    break
                
                # Run detection with IOU threshold to reduce overlapping boxes
                results = self.model(
                    frame, 
                    conf=self.conf_threshold, 
                    iou=0.3,  # Lower IOU = stricter box filtering (less overlap)
                    device=self.device, 
                    verbose=False
                )
                
                # Draw boxes manually with simplified names
                annotated_frame = frame.copy()
                
                for box in results[0].boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class info
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    display_name = self.display_names[class_id]
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    label = f"{display_name} {confidence:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Calculate FPS using exponential moving average for smoother display
                frame_time = time.time() - frame_start
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                fps = fps * 0.9 + current_fps * 0.1 if fps > 0 else current_fps
                fps_counter += 1
                
                # Add info text
                info_text = [
                    f"FPS: {fps:.1f}",
                    f"Confidence: {self.conf_threshold:.2f}",
                    f"Detections: {len(results[0].boxes)}"
                ]
                
                y_offset = 30
                for text in info_text:
                    cv2.putText(annotated_frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30
                
                # Display detected objects
                if len(results[0].boxes) > 0:
                    detections = {}
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        display_name = self.display_names[class_id]
                        confidence = float(box.conf[0])
                        
                        if display_name not in detections:
                            detections[display_name] = []
                        detections[display_name].append(confidence)
                    
                    # Print to console
                    detection_str = " | ".join([f"{cls}: {len(confs)}" for cls, confs in detections.items()])
                    print(f"\rDetected: {detection_str}" + " "*20, end="", flush=True)
                else:
                    print(f"\rNo waste detected" + " "*40, end="", flush=True)
                
                # Show frame
                cv2.imshow(window_name, annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n\nStopping camera...")
                    break
                elif key == ord('s'):
                    screenshot_count += 1
                    filename = f'screenshot_{screenshot_count}.jpg'
                    cv2.imwrite(filename, annotated_frame)
                    print(f"\nScreenshot saved: {filename}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            print("\nCamera released and windows closed")
            print("Detection session completed!")


def main():
    """Main function to run the detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time waste detection using webcam')
    parser.add_argument('--model', type=str, 
                       default='best.pt',
                       help='Path to trained model weights')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (0-1, default: 0.5)')
    
    args = parser.parse_args()
    
    try:
        # Create detector
        detector = WasteDetector(model_path=args.model, conf_threshold=args.conf)
        
        # Run detection
        detector.detect_from_camera(camera_id=args.camera)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo train the model, run:")
        print("  python train.py")
    except RuntimeError as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  - Make sure your camera is connected")
        print("  - Check if another app is using the camera")
        print("  - Try a different camera ID: --camera 1")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
