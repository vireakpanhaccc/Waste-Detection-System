"""
Waste Detection System - Training Script
This script trains a YOLOv8 model for waste classification
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path

def check_gpu():
    """Check if GPU is available and display information"""
    print("=" * 50)
    print("HARDWARE CHECK")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"✓ GPU is available!")
        print(f"  - GPU Count: {torch.cuda.device_count()}")
        print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - PyTorch Version: {torch.__version__}")
        device = 'cuda'
    else:
        print("⚠ No GPU detected - training will use CPU (slower)")
        print(f"  - PyTorch Version: {torch.__version__}")
        device = 'cpu'
    
    print("=" * 50)
    print()
    return device

def train_model(
    model_size='s',  # Options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
    epochs=100,
    imgsz=640,
    batch=16,
    device=None,
    data_yaml='data.yaml',
    project='runs/detect',
    name='waste_detection',
    patience=50
):
    """
    Train YOLOv8 model for waste detection
    
    Args:
        model_size: Size of YOLOv8 model ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size (adjust based on GPU memory)
        device: Device to use ('cuda' or 'cpu', None for auto-detect)
        data_yaml: Path to data.yaml file
        project: Project directory to save results
        name: Name of the training run
    """
    
    # Auto-detect device if not specified
    if device is None:
        device = check_gpu()
    else:
        print(f"Using specified device: {device}")
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data configuration file not found: {data_yaml}")
    
    print(f"Loading YOLOv8{model_size} model...")
    model = YOLO(f'yolov8{model_size}.pt')  # Load pretrained model
    
    print("\n" + "=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"Model: YOLOv8{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch}")
    print(f"Device: {device}")
    print(f"Data Config: {data_yaml}")
    print(f"Output: {project}/{name}")
    print("=" * 50)
    print()
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=50,  # Early stopping patience
        save=True,  # Save checkpoints
        save_period=10,  # Save checkpoint every 10 epochs
        verbose=True,
        plots=True,  # Generate training plots
        # Augmentation parameters (you can adjust these)
        hsv_h=0.015,  # Image HSV-Hue augmentation
        hsv_s=0.7,    # Image HSV-Saturation augmentation
        hsv_v=0.4,    # Image HSV-Value augmentation
        degrees=0.0,  # Image rotation (+/- deg)
        translate=0.1,  # Image translation (+/- fraction)
        scale=0.5,    # Image scale (+/- gain)
        shear=0.0,    # Image shear (+/- deg)
        perspective=0.0,  # Image perspective (+/- fraction)
        flipud=0.0,   # Image flip up-down (probability)
        fliplr=0.5,   # Image flip left-right (probability)
        mosaic=1.0,   # Image mosaic (probability)
    )
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Results saved to: {project}/{name}")
    print(f"Best model: {project}/{name}/weights/best.pt")
    print(f"Last model: {project}/{name}/weights/last.pt")
    print("=" * 50)
    
    return results

def validate_model(model_path, data_yaml='data.yaml', device=None):
    """
    Validate trained model
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to data.yaml file
        device: Device to use for validation
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nValidating model: {model_path}")
    model = YOLO(model_path)
    results = model.val(data=data_yaml, device=device)
    
    print("\nValidation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train waste detection model')
    parser.add_argument('--config', type=str, default='standard',
                       choices=['quick', 'standard', 'high_accuracy', 'max', 'cpu', 'realtime', 'custom'],
                       help='Training configuration preset')
    parser.add_argument('--model-size', type=str, choices=['n', 's', 'm', 'l', 'x'],
                       help='Override model size')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch', type=int, help='Override batch size')
    
    args = parser.parse_args()
    
    # Import configuration
    try:
        from config import get_config
        config = get_config(args.config)
        print(f"Using '{args.config}' configuration")
    except ImportError:
        # Fallback if config.py doesn't exist
        config = {
            'model_size': 'n',
            'epochs': 100,
            'imgsz': 640,
            'batch': 16,
            'data_yaml': 'data.yaml',
            'project': 'runs/detect',
            'name': 'waste_detection_v1'
        }
        print("Using default configuration")
    
    # Apply command line overrides
    if args.model_size:
        config['model_size'] = args.model_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch:
        config['batch'] = args.batch
    
    # Train the model
    results = train_model(**config)
    
    # Optional: Validate the best model
    best_model_path = f"{config['project']}/{config['name']}/weights/best.pt"
    if os.path.exists(best_model_path):
        validate_model(best_model_path, config['data_yaml'])
