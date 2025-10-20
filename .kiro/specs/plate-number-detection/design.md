# Design Document: License Plate Detection System

## Overview

This system implements a license plate detection solution using YOLOv5n, a lightweight object detection model. The architecture follows a modular design with clear separation between model management, image processing, detection logic, and visualization components.

The system will be implemented in Python using PyTorch and the official YOLOv5 repository, providing a simple API for detecting license plates in images with configurable confidence thresholds.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                      │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              PlateDetector (Main Interface)              │
│  - load_model()                                          │
│  - detect(image_path, confidence_threshold)              │
│  - visualize_results(image, detections, output_path)     │
└───────┬─────────────────────────┬───────────────────────┘
        │                         │
        ▼                         ▼
┌──────────────────┐    ┌─────────────────────────┐
│  ModelManager    │    │   ImageProcessor        │
│  - load_yolov5n  │    │   - load_image          │
│  - validate      │    │   - preprocess          │
└──────────────────┘    │   - validate_format     │
                        └─────────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────────────┐
                        │  DetectionEngine        │
                        │  - run_inference        │
                        │  - filter_by_confidence │
                        │  - format_results       │
                        └─────────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────────────┐
                        │  Visualizer             │
                        │  - draw_boxes           │
                        │  - add_labels           │
                        │  - save_image           │
                        └─────────────────────────┘
```

## Components and Interfaces

### 1. PlateDetector (Main Interface)

The primary class that users interact with.

```python
class PlateDetector:
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize the plate detector
        
        Args:
            model_path: Path to custom YOLOv5n weights, or None to use pretrained
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        
    def detect(self, 
               image_path: str, 
               confidence_threshold: float = 0.25) -> List[Detection]:
        """
        Detect license plates in an image
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence score (0-1)
            
        Returns:
            List of Detection objects containing bbox coordinates and scores
        """
        
    def visualize(self, 
                  image_path: str, 
                  detections: List[Detection], 
                  output_path: str) -> None:
        """
        Draw bounding boxes on image and save
        
        Args:
            image_path: Path to original image
            detections: List of Detection objects
            output_path: Where to save annotated image
        """
```

### 2. ModelManager

Handles YOLOv5n model loading and validation.

```python
class ModelManager:
    @staticmethod
    def load_model(model_path: Optional[str], device: str) -> torch.nn.Module:
        """
        Load YOLOv5n model from path or download pretrained
        
        Returns:
            Loaded PyTorch model ready for inference
        """
        
    @staticmethod
    def validate_model(model: torch.nn.Module) -> bool:
        """
        Verify model is properly configured
        """
```

### 3. ImageProcessor

Handles image loading, validation, and preprocessing.

```python
class ImageProcessor:
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Load and validate image file
        
        Raises:
            ValueError: If image is invalid or corrupted
        """
        
    @staticmethod
    def preprocess(image: np.ndarray, target_size: int = 640) -> torch.Tensor:
        """
        Preprocess image for YOLOv5n input
        - Resize to target size
        - Normalize pixel values
        - Convert to tensor
        """
```

### 4. DetectionEngine

Runs inference and processes results.

```python
class DetectionEngine:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        
    def run_inference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run model inference on preprocessed image
        """
        
    def filter_by_confidence(self, 
                            raw_detections: torch.Tensor, 
                            threshold: float) -> List[Detection]:
        """
        Filter detections by confidence threshold and format results
        """
```

### 5. Visualizer

Handles result visualization.

```python
class Visualizer:
    @staticmethod
    def draw_boxes(image: np.ndarray, 
                   detections: List[Detection]) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        """
        
    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> None:
        """
        Save annotated image to disk
        """
```

## Data Models

### Detection

```python
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float                 # 0.0 to 1.0
    class_name: str                   # "license_plate"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        
    def area(self) -> int:
        """Calculate bounding box area"""
```

### Configuration

```python
@dataclass
class DetectorConfig:
    model_path: Optional[str] = None
    device: str = 'cpu'
    default_confidence: float = 0.25
    image_size: int = 640
    supported_formats: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp'])
```

## Error Handling

### Error Types

1. **ModelLoadError**: Raised when model fails to load
   - Invalid model path
   - Corrupted weights file
   - Incompatible model format

2. **ImageProcessingError**: Raised for image-related issues
   - Unsupported format
   - Corrupted image file
   - Invalid file path

3. **InferenceError**: Raised during detection
   - Model inference failure
   - Out of memory errors
   - Invalid input dimensions

### Error Handling Strategy

```python
try:
    detector = PlateDetector()
    detections = detector.detect("image.jpg", confidence_threshold=0.3)
except ModelLoadError as e:
    logger.error(f"Failed to load model: {e}")
    # Fallback or retry logic
except ImageProcessingError as e:
    logger.error(f"Image processing failed: {e}")
    # Skip image or request valid input
except InferenceError as e:
    logger.error(f"Detection failed: {e}")
    # Return empty results or retry
```

All errors will include descriptive messages and will not cause system crashes. The system will log errors and return appropriate error responses.

## Testing Strategy

### Unit Tests

1. **ModelManager Tests**
   - Test model loading with valid/invalid paths
   - Test automatic download functionality
   - Test model validation

2. **ImageProcessor Tests**
   - Test loading various image formats
   - Test preprocessing output dimensions
   - Test error handling for corrupted images

3. **DetectionEngine Tests**
   - Test inference with mock model
   - Test confidence filtering
   - Test result formatting

4. **Visualizer Tests**
   - Test bounding box drawing
   - Test label rendering
   - Test image saving

### Integration Tests

1. **End-to-End Detection**
   - Test complete detection pipeline with sample images
   - Verify detection accuracy on known test cases
   - Test with various confidence thresholds

2. **Performance Tests**
   - Measure inference time on standard hardware
   - Verify memory usage stays within bounds
   - Test batch processing capabilities

### Test Data

- Sample images with license plates (various angles, lighting conditions)
- Images without license plates (negative cases)
- Corrupted/invalid image files
- Edge cases (very small/large images)

## Implementation Notes

### Dependencies

- Python 3.8+
- PyTorch >= 1.7
- OpenCV (cv2) for image processing
- NumPy for array operations
- YOLOv5 repository (via pip or git clone)

### Model Training Considerations

While this design focuses on using a pretrained or custom YOLOv5n model, the system should support:
- Loading custom-trained weights for license plate detection
- Fine-tuning on specific datasets if needed
- The model should be trained on a license plate dataset (e.g., annotated images with plate bounding boxes)

### Performance Optimization

- Use GPU acceleration when available (CUDA)
- Implement batch processing for multiple images
- Cache model in memory to avoid reloading
- Optimize image preprocessing pipeline

### Future Enhancements

- Support for video stream processing
- Real-time detection with webcam input
- OCR integration to read plate numbers
- Multi-class detection (plates from different regions)
- REST API for remote detection requests
