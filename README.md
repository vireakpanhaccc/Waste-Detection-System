# Waste Detection System

Real-time waste detection and classification system using YOLOv8 for automated waste sorting and environmental monitoring.

## 🎯 Project Overview

This project implements a real-time object detection system that identifies and classifies waste materials commonly found in daily life. The system uses the YOLO (You Only Look Once) algorithm to detect waste items through a webcam feed, enabling automated waste sorting and classification.

### Detected Waste Categories

Our model classifies waste into **14 detailed categories**:

**Plastics (6 types):**
- HDPE (High-Density Polyethylene) - Milk jugs, detergent bottles
- LDPE (Low-Density Polyethylene) - Plastic bags, squeeze bottles  
- PETE (Polyethylene Terephthalate) - Water bottles, food containers
- PP (Polypropylene) - Yogurt containers, bottle caps
- PS (Polystyrene) - Foam cups, takeout containers
- PVC (Polyvinyl Chloride) - Pipes, credit cards

**Other Materials (8 types):**
- **Glass** - Bottles, jars
- **Metal** - Cans, containers
- **Paper** - Documents, newspapers
- **Organic** - Biodegradable materials
- **Battery** - Rechargeable and disposable batteries
- **Electronic** - Circuit boards, small electronics
- **Light Bulb** - Various bulb types
- **Automobile** - Car parts, automotive waste

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/vireakpanhaccc/Waste-Detection-System.git
cd Waste-Detection-System

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Get the Trained Model

**Note:** The trained model (`best.pt`) is not included in the repository due to `.gitignore` settings.

**Option A: Train Your Own Model**
```bash
python train.py
```

**Option B: Download Pre-trained Model**
Download `best.pt` from [releases/shared location] and place it in the project root directory.

### 3. Run Real-Time Detection

```bash
# Basic usage (uses best.pt in current directory)
python camera_detect.py

# Adjust confidence threshold
python camera_detect.py --conf 0.3

# Use different camera
python camera_detect.py --camera 1

# Specify model path
python camera_detect.py --model path/to/your/best.pt
```

## 📁 Project Structure

```
Waste-Detection-System/
├── camera_detect.py           # Real-time webcam detection
├── train.py                   # Model training script
├── evaluate.py                # Model evaluation & metrics
├── predict.py                 # Batch image/video inference
├── test_live.py              # Alternative live testing
├── test_model.py             # Model testing utilities
├── config.py                 # Training configurations
├── data.yaml                 # Dataset configuration
├── requirements.txt          # Python dependencies
├── CAMERA_DETECTION_GUIDE.md # Detailed camera setup guide
├── train/                    # Training images & labels
├── valid/                    # Validation images & labels
├── test/                     # Test images & labels
├── venv/                     # Virtual environment (local)
└── best.pt                   # Trained model (not in git)
```

## 🎮 Camera Detection Controls

While running `camera_detect.py`:

| Key | Action |
|-----|--------|
| **q** | Quit the application |
| **s** | Save screenshot with detections |

## 🔧 Features

### ✅ Real-Time Detection
- Live webcam feed processing
- 15-30+ FPS on CPU, 60+ FPS on GPU
- Confidence threshold: 0.25 (adjustable)
- IOU threshold: 0.3 for reduced overlaps
- Simplified plastic type display

### ✅ Training Pipeline
- Pre-trained YOLOv8 base model
- Custom dataset fine-tuning
- Data augmentation (crop, flip, brightness, contrast)
- Multiple model sizes (n/s/m/l/x)
- Early stopping & checkpointing

### ✅ Evaluation Metrics
- mAP@0.5 and mAP@0.5:0.95
- Per-class precision, recall, F1
- Confusion matrix
- FPS measurement
- Comprehensive performance reports

## 📊 Model Training

### Available Configurations

Edit `config.py` or use command-line arguments:

```python
# Quick training (testing)
QUICK_CONFIG = {
    'model_size': 'n',
    'epochs': 50,
    'batch': 16,
}

# Standard training (recommended)
STANDARD_CONFIG = {
    'model_size': 'n', 
    'epochs': 100,
    'batch': 16,
}

# High accuracy
HIGH_ACCURACY_CONFIG = {
    'model_size': 'm',
    'epochs': 200,
    'batch': 32,
}
```

### Train the Model

```bash
# Use default standard config
python train.py

# Quick test run
python train.py --config quick

# High accuracy training
python train.py --config high_accuracy
```

## 📈 Model Evaluation

```bash
# Evaluate model on test set
python evaluate.py --model best.pt

# Generate detailed reports with visualizations
python evaluate.py --model best.pt --save-plots
```

**Evaluation outputs:**
- mAP scores (0.5 and 0.5:0.95)
- Precision, Recall, F1 per class
- Confusion matrix
- Performance visualizations
- JSON metrics export

## 🛠️ Requirements

### System Requirements
- **Python:** 3.8+ (tested on 3.12)
- **OS:** macOS, Linux, or Windows
- **RAM:** 8GB minimum, 16GB recommended
- **GPU:** Optional but recommended (NVIDIA with CUDA support)

### Python Packages
```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
PyYAML>=6.0
matplotlib>=3.7.0
pandas>=2.0.0
pillow>=10.0.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

## 💡 Usage Tips

### Improving Detection Accuracy

1. **Lighting:** Ensure objects are well-lit, avoid backlighting
2. **Distance:** Keep objects 1-3 feet from camera
3. **Background:** Use plain backgrounds when possible
4. **Angle:** Present objects at similar angles as training data
5. **Confidence:** Lower threshold (0.15-0.25) for more detections

### Performance Optimization

```bash
# Lower confidence for more detections
python camera_detect.py --conf 0.15

# Higher confidence for fewer false positives
python camera_detect.py --conf 0.5
```

## 🔬 Research & Methodology

### Approach
1. **Data Preparation:** Labeled dataset with train/valid/test splits
2. **Preprocessing:** Image augmentation (crop, flip, brightness, contrast)
3. **Training:** Fine-tuned pre-trained YOLOv8 using supervised learning
4. **Evaluation:** Comprehensive metrics (mAP, precision, recall, F1, FPS)
5. **Deployment:** Real-time camera-based detection system

### Reproducibility
- Fixed random seeds for consistent results
- Version-controlled code and configurations
- Documented training parameters
- Evaluation methodology clearly defined

## 📝 Dataset Information

- **Source:** Roboflow Universe
- **Classes:** 14 waste categories
- **Format:** YOLO v8 Detection format
- **Splits:** Train / Validation / Test
- **Annotations:** Bounding boxes with class labels

Dataset URL: https://universe.roboflow.com/aolai-jbgyl/waste-type-ui2bs/dataset/7

## 🐛 Troubleshooting

### Camera won't open
```bash
# Try different camera ID
python camera_detect.py --camera 1

# Check camera permissions in System Preferences (macOS)
```

### Model not found
```bash
# Ensure best.pt is in the project root
ls -la best.pt

# Or specify full path
python camera_detect.py --model /full/path/to/best.pt
```

### Low detection rate
```bash
# Lower confidence threshold
python camera_detect.py --conf 0.15

# Check if objects are in trained categories
python -c "from ultralytics import YOLO; print(YOLO('best.pt').names)"
```

### Poor performance
- Retrain with more diverse dataset
- Ensure good lighting and clear object presentation
- Use GPU for faster processing

## 🙏 Acknowledgments

- **YOLOv8:** Ultralytics - https://github.com/ultralytics/ultralytics
- **Dataset:** Roboflow Universe Community
- **Framework:** PyTorch
- **Computer Vision:** OpenCV

## 📄 License

This project uses a dataset licensed under CC BY 4.0. Please refer to the dataset source for specific licensing terms.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

**Note:** This project is designed for research and educational purposes. For production deployment, additional testing and validation are recommended.
