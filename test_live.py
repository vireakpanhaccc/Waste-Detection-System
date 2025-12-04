"""
Live Webcam Test for Waste Detection System
Opens your camera and detects waste in real-time
"""

import cv2
from ultralytics import YOLO
import torch
import os

def test_live_camera(model_path='runs/detect/waste_detection_standard3/weights/best.pt', 
                     conf_threshold=0.25,
                     camera_id=0):
    """
    Run real-time waste detection using webcam
    
    Args:
        model_path: Path to trained model weights
        conf_threshold: Confidence threshold for detections
        camera_id: Camera device ID (0 for default webcam)
    """
    
    print("=" * 70)
    print("LIVE WASTE DETECTION - WEBCAM TEST")
    print("=" * 70)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📱 Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        print("   Please train the model first:")
        print("   ./run_train.sh --config quick")
        return
    
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"🎯 Confidence threshold: {conf_threshold}")
    
    # Display detectable waste types
    print(f"\n🗑️  Detectable waste types:")
    for idx, name in model.names.items():
        print(f"   {idx}: {name}")
    
    print("\n" + "=" * 70)
    print("STARTING CAMERA...")
    print("=" * 70)
    print("\n📹 Opening camera (press 'q' to quit)...")
    print("   Press 'q' to quit")
    print("   Press 's' to save screenshot")
    print("   Press '+' to increase confidence threshold")
    print("   Press '-' to decrease confidence threshold")
    
    # Run inference on webcam
    try:
        results = model.predict(
            source=camera_id,
            conf=conf_threshold,
            show=True,  # Display window
            device=device,
            stream=True,  # Stream results
            verbose=False
        )
        
        screenshot_count = 0
        current_conf = conf_threshold
        
        for r in results:
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Stopping camera...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f'screenshot_{screenshot_count}.jpg'
                cv2.imwrite(filename, r.plot())
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('+') or key == ord('='):
                current_conf = min(0.95, current_conf + 0.05)
                print(f"🎯 Confidence threshold: {current_conf:.2f}")
                model.conf = current_conf
            elif key == ord('-'):
                current_conf = max(0.05, current_conf - 0.05)
                print(f"🎯 Confidence threshold: {current_conf:.2f}")
                model.conf = current_conf
            
            # Show detection info
            num_detections = len(r.boxes)
            if num_detections > 0:
                detections = {}
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    if class_name not in detections:
                        detections[class_name] = []
                    detections[class_name].append(confidence)
                
                # Print detections
                print(f"\r🔍 Detected: ", end="")
                for cls, confs in detections.items():
                    avg_conf = sum(confs) / len(confs)
                    print(f"{cls}({len(confs)}, {avg_conf:.0%}) ", end="")
                print(" " * 20, end="")  # Clear line
            else:
                print(f"\r⚪ No waste detected" + " " * 40, end="")
        
        cv2.destroyAllWindows()
        print("\n\n✅ Camera test completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  - Make sure your camera is connected")
        print("  - Check if another app is using the camera")
        print("  - Try a different camera ID: ./run_live_test.sh --camera 1")

def test_video_file(video_path, model_path='runs/detect/waste_detection_standard3/weights/best.pt', 
                    conf_threshold=0.25):
    """
    Run waste detection on a video file
    
    Args:
        video_path: Path to video file
        model_path: Path to trained model
        conf_threshold: Confidence threshold
    """
    
    print("=" * 70)
    print("VIDEO FILE WASTE DETECTION TEST")
    print("=" * 70)
    
    if not os.path.exists(video_path):
        print(f"\n❌ Video not found: {video_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📱 Device: {device}")
    print(f"🎬 Video: {video_path}")
    print(f"📦 Model: {model_path}")
    
    model = YOLO(model_path)
    
    print("\n📹 Processing video (press 'q' to quit)...")
    
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        save=True,
        project='test_results',
        name='video_test',
        device=device,
        stream=True,
        verbose=False
    )
    
    frame_count = 0
    total_detections = 0
    
    for r in results:
        frame_count += 1
        total_detections += len(r.boxes)
        
        if frame_count % 30 == 0:  # Print every 30 frames (~1 sec)
            print(f"📊 Processed {frame_count} frames, {total_detections} total detections")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"\n✅ Processed {frame_count} frames")
    print(f"✅ Total detections: {total_detections}")
    print(f"✅ Output saved to: test_results/video_test/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Live waste detection test')
    parser.add_argument('--model', type=str, 
                       default='runs/detect/waste_detection_standard3/weights/best.pt',
                       help='Path to trained model weights')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file (if not using webcam)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    
    args = parser.parse_args()
    
    if args.video:
        # Test video file
        test_video_file(args.video, args.model, args.conf)
    else:
        # Test live camera
        test_live_camera(args.model, args.conf, args.camera)
