# Waste Detection System

A YOLOv8-based object detection system for classifying different types of waste materials.

## 🎯 Project Overview

This system can detect and classify 6 types of waste:
- **Metal** - Cans, metal containers
- **Paper** - Documents, cardboard boxes, newspapers
- **Plastic** - Bottles, bags, containers
- **Random Trash** - Mixed/unclassifiable waste
- **Cardboard** - Boxes, packaging materials
- **Glass** - Bottles, jars, glass containers

## 🚀 Quick Start

### Train the model
\`\`\`bash
python train.py
\`\`\`

### Evaluate performance
\`\`\`bash
python evaluate.py
\`\`\`

### Run predictions
\`\`\`bash
python predict.py --source test/images/
\`\`\`

## 📊 Project Structure

\`\`\`
Waste-Detection-System/
├── train.py           # Training script
├── evaluate.py        # Model evaluation & metrics
├── predict.py         # Inference on images/video/webcam
├── requirements.txt   # Dependencies
├── data.yaml         # Dataset configuration
├── train/            # Training dataset
├── valid/            # Validation dataset
├── test/             # Test dataset
└── runs/             # Training outputs (created automatically)
\`\`\`

## 🔧 Features

### ✅ Training (`train.py`)
- Automatic GPU/CPU detection
- Multiple model sizes (nano to xlarge)
- Data augmentation
- Checkpoint saving
- Early stopping
- Training visualization

### ✅ Evaluation (`evaluate.py`)
- Comprehensive metrics (mAP, precision, recall)
- Per-class performance analysis
- Confusion matrix
- Performance visualization plots
- JSON metrics export

### ✅ Inference (`predict.py`)
- Single image prediction
- Batch folder processing
- Video file processing
- Real-time webcam detection
- Confidence threshold control

## 📈 Usage Examples

### Training with custom parameters
\`\`\`python
# Edit train.py config:
config = {
    'model_size': 's',    # Use small model
    'epochs': 150,        # Train longer
    'batch': 32,          # Larger batch (if GPU allows)
}
\`\`\`

### Evaluation with plots
\`\`\`bash
python evaluate.py --model runs/detect/waste_detection_v1/weights/best.pt --test-images
\`\`\`

### Real-time webcam detection
\`\`\`bash
python predict.py --source webcam --conf 0.5
\`\`\`

## 🎓 Model Performance Tips

| Scenario | Recommendation |
|----------|----------------|
| **Fast training** | model='n', batch=16, epochs=100 |
| **Best accuracy** | model='l', batch=32, epochs=200 |
| **Limited GPU** | model='n', batch=8, reduce imgsz |
| **Real-time app** | Use YOLOv8n model |

## 🛠️ Installation

\`\`\`bash
# Clone repository
git clone <your-repo>
cd Waste-Detection-System

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU (optional)
python -c "import torch; print(torch.cuda.is_available())"
\`\`\`

## 📦 Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA 12.1 (for GPU)
- 8GB+ RAM (16GB recommended)
- NVIDIA GPU with 4GB+ VRAM (recommended)

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Dataset from Roboflow Universe
