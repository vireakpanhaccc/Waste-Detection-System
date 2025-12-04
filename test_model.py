"""
Test Script for Waste Detection System
Tests the trained model with sample images and displays results
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch

def test_model(model_path='/home/paul/Waste-Detection-System/runs/detect/waste_detection_standard/weights/best.pt', 
               test_images_dir='test/images',
               conf_threshold=0.25,
               num_samples=10):
    """
    Test the waste detection model on sample images
    
    Args:
        model_path: Path to trained model weights
        test_images_dir: Directory containing test images
        conf_threshold: Confidence threshold for detections
        num_samples: Number of sample images to test
    """
    
    print("=" * 70)
    print("WASTE DETECTION MODEL - TEST RUN")
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
    
    # Check test images
    test_dir = Path(test_images_dir)
    if not test_dir.exists():
        print(f"\n❌ Test directory not found: {test_images_dir}")
        return
    
    # Get sample images
    image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
    
    if not image_files:
        print(f"\n❌ No images found in {test_images_dir}")
        return
    
    # Limit to num_samples
    image_files = image_files[:num_samples]
    
    print(f"\n🖼️  Found {len(image_files)} test images")
    print(f"   Testing {len(image_files)} samples with confidence threshold: {conf_threshold}")
    print("=" * 70)
    
    # Class names
    class_names = model.names
    print(f"\n📋 Detectable waste types:")
    for idx, name in class_names.items():
        print(f"   {idx}: {name}")
    
    print("\n" + "=" * 70)
    print("RUNNING PREDICTIONS")
    print("=" * 70)
    
    # Run predictions
    results = model.predict(
        source=image_files,
        conf=conf_threshold,
        save=True,
        save_txt=True,
        save_conf=True,
        project='test_results',
        name='predictions',
        device=device,
        verbose=False
    )
    
    # Analyze results
    total_detections = 0
    detection_counts = {name: 0 for name in class_names.values()}
    images_with_detections = 0
    
    print("\n📊 DETECTION RESULTS:")
    print("-" * 70)
    
    for i, (result, img_path) in enumerate(zip(results, image_files), 1):
        num_detections = len(result.boxes)
        total_detections += num_detections
        
        if num_detections > 0:
            images_with_detections += 1
        
        print(f"\n{i}. {img_path.name}")
        
        if num_detections == 0:
            print("   ❌ No waste detected")
        else:
            print(f"   ✓ Found {num_detections} object(s):")
            
            # Group detections by class
            detections_by_class = {}
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = class_names[class_id]
                
                if class_name not in detections_by_class:
                    detections_by_class[class_name] = []
                detections_by_class[class_name].append(confidence)
                detection_counts[class_name] += 1
            
            # Display grouped detections
            for class_name, confidences in detections_by_class.items():
                avg_conf = sum(confidences) / len(confidences)
                print(f"      - {class_name}: {len(confidences)}x (avg confidence: {avg_conf:.2%})")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\n📈 Overall Results:")
    print(f"   Images tested: {len(image_files)}")
    print(f"   Images with detections: {images_with_detections} ({images_with_detections/len(image_files)*100:.1f}%)")
    print(f"   Total objects detected: {total_detections}")
    print(f"   Average detections per image: {total_detections/len(image_files):.2f}")
    
    print(f"\n🗑️  Detections by waste type:")
    for class_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            print(f"   {class_name:15s}: {count:3d} ({percentage:5.1f}%)")
    
    print("\n" + "=" * 70)
    print(f"✅ Predictions saved to: test_results/predictions/")
    print(f"   - Annotated images with bounding boxes")
    print(f"   - Labels in YOLO format (.txt files)")
    print("=" * 70)
    
    return results

def test_single_image(image_path, model_path='runs/detect/waste_detection_standard3/weights/best.pt', conf_threshold=0.25):
    """
    Test on a single image and display detailed results
    
    Args:
        image_path: Path to image file
        model_path: Path to trained model
        conf_threshold: Confidence threshold
    """
    
    print("=" * 70)
    print("SINGLE IMAGE TEST")
    print("=" * 70)
    
    if not os.path.exists(image_path):
        print(f"\n❌ Image not found: {image_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📱 Device: {device}")
    print(f"🖼️  Image: {image_path}")
    print(f"📦 Model: {model_path}")
    print(f"🎯 Confidence threshold: {conf_threshold}")
    
    model = YOLO(model_path)
    
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        project='test_results',
        name='single_test',
        device=device,
        show_labels=True,
        show_conf=True
    )
    
    result = results[0]
    
    print("\n" + "=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)
    
    if len(result.boxes) == 0:
        print("\n❌ No waste detected in this image")
    else:
        print(f"\n✅ Detected {len(result.boxes)} object(s):\n")
        
        for i, box in enumerate(result.boxes, 1):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            print(f"{i}. {class_name.upper()}")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   Bounding box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
            print()
    
    print("=" * 70)
    print(f"✅ Result saved to: test_results/single_test/")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test waste detection model')
    parser.add_argument('--model', type=str, 
                       default='runs/detect/waste_detection_standard3/weights/best.pt',
                       help='Path to trained model weights')
    parser.add_argument('--image', type=str, default=None,
                       help='Test single image (path to image file)')
    parser.add_argument('--test-dir', type=str, default='test/images',
                       help='Directory with test images')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of sample images to test')
    
    args = parser.parse_args()
    
    if args.image:
        # Test single image
        test_single_image(args.image, args.model, args.conf)
    else:
        # Test multiple images
        test_model(args.model, args.test_dir, args.conf, args.samples)
