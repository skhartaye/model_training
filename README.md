# Philippine License Plate Detection using YOLOv5n

A custom-trained YOLOv5n model for detecting Philippine license plates with high accuracy.

## 🎯 Model Performance

- **mAP50**: 82.6% (overall accuracy)
- **License Plate Detection**: 92.2% mAP50
- **Plate Content Detection**: 73.1% mAP50
- **Training Time**: ~1 hour on CPU
- **Model Size**: 3.8MB (YOLOv5n - nano variant)

## 📋 Requirements

- Python 3.8+
- PyTorch >= 1.7
- OpenCV
- NumPy
- YOLOv5

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/skhartaye/model_training.git
cd model_training
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the Philippine License Plate dataset from Roboflow or your source and place it in the project directory.

### 4. Train the Model

```bash
python train_philippine_plates.py
```

The script will:
- Clone YOLOv5 repository
- Convert OBB labels to YOLO format
- Train YOLOv5n model
- Save results to `runs/train/philippine_plates/`

### 5. Test the Model

```bash
python yolov5/detect.py --weights runs/train/philippine_plates2/weights/best.pt --source path/to/test/images --conf 0.3 --save-txt --save-conf
```

Results will be saved to `yolov5/runs/detect/exp/`

## 📁 Project Structure

```
model_training/
├── train_philippine_plates.py    # Training script
├── test_trained_model.py          # Testing script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── Philippine_License_Plate.v1i.yolov5-obb/  # Dataset (not included)
│   ├── train/
│   ├── valid/
│   └── test/
└── runs/                          # Training outputs
    └── train/
        └── philippine_plates/
            └── weights/
                ├── best.pt        # Best model weights
                └── last.pt        # Last epoch weights
```

## 🎓 Training Details

### Dataset
- **Source**: Roboflow Philippine License Plate Dataset
- **Format**: YOLOv5 OBB (Oriented Bounding Box)
- **Classes**: 3
  - Invalid Plate
  - License Plate
  - Plate Content

### Training Configuration
- **Model**: YOLOv5n (nano)
- **Epochs**: 100
- **Batch Size**: 16
- **Image Size**: 640x640
- **Device**: CPU (GPU recommended for faster training)

### Data Preprocessing
The training script automatically:
1. Converts OBB format to standard YOLO format
2. Normalizes bounding box coordinates
3. Validates image dimensions
4. Creates train/val/test splits

## 🧪 Testing

### Option 1: Using YOLOv5 Detect Script (Recommended)

```bash
python yolov5/detect.py \
  --weights runs/train/philippine_plates2/weights/best.pt \
  --source Philippine_License_Plate.v1i.yolov5-obb/test/images \
  --conf 0.3 \
  --save-txt \
  --save-conf
```

### Option 2: Using Test Script

```bash
python test_trained_model.py
```

## 📊 Results

The trained model achieves:
- **82.6% mAP50** overall
- **92.2% accuracy** on license plate detection
- **73.1% accuracy** on plate content detection

Results include:
- Annotated images with bounding boxes
- Detection confidence scores
- Label files with coordinates

## 🔧 Customization

### Adjust Confidence Threshold

```bash
python yolov5/detect.py --weights best.pt --source images/ --conf 0.5
```

### Train with Different Parameters

Edit `train_philippine_plates.py`:
```python
epochs = 150  # Increase epochs
batch_size = 32  # Increase batch size (requires more memory)
img_size = 1280  # Increase image size for better accuracy
```

## 📝 Notes

- The model is trained specifically for Philippine license plates
- Works best with clear, front-facing plate images
- GPU training is recommended for faster results
- Model weights are not included in the repository (too large for GitHub)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [Roboflow](https://roboflow.com) for dataset tools
- Philippine License Plate Dataset contributors

## 📧 Contact

For questions or issues, please open an issue on GitHub.
