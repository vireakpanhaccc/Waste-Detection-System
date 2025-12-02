"""
Model Evaluation Script
Comprehensive evaluation of trained waste detection model
"""

from ultralytics import YOLO
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

def evaluate_model(model_path, data_yaml='data.yaml', save_dir='evaluation_results'):
    """
    Comprehensive model evaluation
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to data configuration
        save_dir: Directory to save evaluation results
    """
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    model = YOLO(model_path)
    
    # Run validation
    print("\n" + "="*60)
    print("RUNNING VALIDATION")
    print("="*60)
    results = model.val(data=data_yaml, device=device, save_json=True, plots=True)
    
    # Extract metrics
    metrics = {
        'timestamp': timestamp,
        'model_path': str(model_path),
        'device': device,
        'metrics': {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
        },
        'per_class_metrics': {}
    }
    
    # Per-class metrics
    class_names = model.names
    if hasattr(results.box, 'maps') and results.box.maps is not None:
        for i, class_name in class_names.items():
            metrics['per_class_metrics'][class_name] = {
                'mAP50-95': float(results.box.maps[i]) if i < len(results.box.maps) else 0.0
            }
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Metrics:")
    print(f"  mAP50:     {metrics['metrics']['mAP50']:.4f}")
    print(f"  mAP50-95:  {metrics['metrics']['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['metrics']['precision']:.4f}")
    print(f"  Recall:    {metrics['metrics']['recall']:.4f}")
    
    print(f"\nPer-Class Performance:")
    for class_name, class_metrics in metrics['per_class_metrics'].items():
        print(f"  {class_name:15s}: mAP50-95 = {class_metrics['mAP50-95']:.4f}")
    
    # Save metrics to JSON
    metrics_file = save_dir / f'metrics_{timestamp}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_file}")
    
    # Generate confusion matrix plot if available
    try:
        create_performance_plots(metrics, save_dir, timestamp)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    print("="*60)
    return metrics

def create_performance_plots(metrics, save_dir, timestamp):
    """Create visualization plots for model performance"""
    
    # Class performance bar chart
    if metrics['per_class_metrics']:
        fig, ax = plt.subplots(figsize=(10, 6))
        classes = list(metrics['per_class_metrics'].keys())
        maps = [m['mAP50-95'] for m in metrics['per_class_metrics'].values()]
        
        bars = ax.bar(classes, maps, color='steelblue', alpha=0.8)
        ax.set_ylabel('mAP50-95', fontsize=12)
        ax.set_xlabel('Waste Class', fontsize=12)
        ax.set_title('Per-Class Performance', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_dir / f'class_performance_{timestamp}.png', dpi=300)
        plt.close()
        print(f"✓ Class performance plot saved")
    
    # Overall metrics comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    metric_names = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
    metric_values = [
        metrics['metrics']['mAP50'],
        metrics['metrics']['mAP50-95'],
        metrics['metrics']['precision'],
        metrics['metrics']['recall']
    ]
    
    bars = ax.barh(metric_names, metric_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Overall Model Performance', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'overall_metrics_{timestamp}.png', dpi=300)
    plt.close()
    print(f"✓ Overall metrics plot saved")

def test_model_predictions(model_path, test_images_dir='test/images', save_dir='test_predictions'):
    """
    Run inference on test images and save predictions
    
    Args:
        model_path: Path to trained model
        test_images_dir: Directory containing test images
        save_dir: Directory to save predictions
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    
    test_images = Path(test_images_dir)
    if not test_images.exists():
        print(f"Warning: Test images directory not found: {test_images_dir}")
        return
    
    image_files = list(test_images.glob('*.jpg')) + list(test_images.glob('*.png'))
    
    if not image_files:
        print(f"No images found in {test_images_dir}")
        return
    
    print(f"\nRunning predictions on {len(image_files)} test images...")
    
    results = model.predict(
        source=str(test_images),
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(save_dir),
        name='predictions',
        device=device,
        conf=0.25
    )
    
    print(f"✓ Predictions saved to: {save_dir}/predictions")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate waste detection model')
    parser.add_argument('--model', type=str, default='runs/detect/waste_detection_v1/weights/best.pt',
                       help='Path to model weights')
    parser.add_argument('--data', type=str, default='data.yaml',
                       help='Path to data.yaml')
    parser.add_argument('--save-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--test-images', action='store_true',
                       help='Also run predictions on test images')
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate_model(args.model, args.data, args.save_dir)
    
    # Optional: test predictions
    if args.test_images:
        test_model_predictions(args.model)
