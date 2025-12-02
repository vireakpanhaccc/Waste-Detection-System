# Quick Reference Guide - Waste Detection System

## 🚀 Quick Commands

### Training
```bash
# Standard training (recommended for first run)
python train.py

# Quick test training (50 epochs)
python train.py --config quick

# High accuracy training (requires good GPU)
python train.py --config high_accuracy

# CPU-only training
python train.py --config cpu

# Custom overrides
python train.py --config standard --epochs 150 --batch 32
```

### Evaluation
```bash
# Evaluate trained model
python evaluate.py --model runs/detect/waste_detection_standard/weights/best.pt

# Include test predictions
python evaluate.py --model runs/detect/waste_detection_standard/weights/best.pt --test-images
```

### Inference
```bash
# Single image
python predict.py --source path/to/image.jpg

# Folder of images
python predict.py --source test/images/

# Video file
python predict.py --source video.mp4

# Webcam (real-time)
python predict.py --source webcam

# With higher confidence threshold
python predict.py --source test/images/ --conf 0.5
```

### View Configurations
```bash
# See all available training configs
python config.py
```

## 📊 Expected Outputs

### After Training
```
runs/detect/waste_detection_standard/
├── weights/
│   ├── best.pt          # Best model checkpoint
│   └── last.pt          # Last epoch checkpoint
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
├── F1_curve.png        # F1 score curve
├── PR_curve.png        # Precision-Recall curve
└── args.yaml           # Training arguments
```

### After Evaluation
```
evaluation_results/
├── metrics_YYYYMMDD_HHMMSS.json          # JSON metrics
├── class_performance_YYYYMMDD_HHMMSS.png # Per-class chart
└── overall_metrics_YYYYMMDD_HHMMSS.png   # Overall metrics
```

### After Prediction
```
predictions/results/
├── image1.jpg           # Annotated images
├── image2.jpg
└── labels/
    ├── image1.txt      # YOLO format detections
    └── image2.txt
```

## 🎯 Project Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script with GPU detection |
| `evaluate.py` | Model evaluation with metrics & plots |
| `predict.py` | Inference on images/video/webcam |
| `config.py` | Training configuration presets |
| `requirements.txt` | Python dependencies |
| `data.yaml` | Dataset configuration |
| `.gitignore` | Git ignore rules |
| `README.md` | Full documentation |

## 💡 Configuration Presets

| Preset | Model | Epochs | Batch | Use Case |
|--------|-------|--------|-------|----------|
| `quick` | n | 50 | 16 | Testing, debugging |
| `standard` | n | 100 | 16 | Default training |
| `high_accuracy` | m | 200 | 32 | Better performance |
| `max` | l | 300 | 32 | Best accuracy |
| `cpu` | n | 50 | 2 | CPU-only training |
| `realtime` | n | 100 | 16 | Fast inference |

## 🔧 Common Adjustments

### Increase Training Time
```python
python train.py --epochs 200
```

### Use Larger Model
```python
python train.py --model-size m
```

### Reduce Memory Usage
```python
python train.py --batch 8
```

### Change Confidence Threshold
```python
python predict.py --source test/images/ --conf 0.3
```

## 📈 Monitoring Training

Watch training in real-time:
```bash
# Terminal 1: Start training
python train.py

# Terminal 2: Monitor GPU usage (if using GPU)
watch -n 1 nvidia-smi
```

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Check GPU
nvidia-smi

# Verify PyTorch sees GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Out of Memory
```bash
# Reduce batch size
python train.py --batch 8

# Or use smaller model
python train.py --config cpu
```

### Training Too Slow
- Use GPU instead of CPU
- Reduce image size in config.py
- Use smaller model (n instead of m/l)

## 📚 Next Steps

1. ✅ **First Training Run**: `python train.py --config quick`
2. ✅ **Evaluate Results**: `python evaluate.py`
3. ✅ **Test Predictions**: `python predict.py --source test/images/`
4. ✅ **Full Training**: `python train.py --config standard`
5. ✅ **Deploy Model**: Use best.pt for your application

## 🎓 Model Selection Guide

- **YOLOv8n**: Fastest, smallest, good for edge devices
- **YOLOv8s**: Balanced speed/accuracy
- **YOLOv8m**: High accuracy, moderate speed
- **YOLOv8l**: Very high accuracy, slower
- **YOLOv8x**: Maximum accuracy, slowest

Choose based on your deployment needs!
