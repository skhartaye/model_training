"""
Test the trained Philippine license plate model using YOLOv5 directly
"""

import torch
from pathlib import Path

# Load your trained model
model_path = "runs/train/philippine_plates2/weights/best.pt"
print(f"Loading model from: {model_path}")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Set confidence threshold
model.conf = 0.3

# Test on images
test_dir = Path('Philippine_License_Plate.v1i.yolov5-obb/test/images')
output_dir = Path('test_results')
output_dir.mkdir(exist_ok=True)

test_images = list(test_dir.glob('*.jpg'))[:5]  # Test first 5 images

print(f"\nTesting on {len(test_images)} images...\n")

for img_path in test_images:
    # Run inference
    results = model(str(img_path))
    
    # Get detections
    detections = results.pandas().xyxy[0]
    
    print(f"{img_path.name}: {len(detections)} detections")
    
    for idx, row in detections.iterrows():
        print(f"  - {row['name']}: {row['confidence']:.2f}")
    
    # Save results with bounding boxes
    output_path = output_dir / f"result_{img_path.name}"
    results.save(save_dir=str(output_dir))
    print(f"  Saved to: {output_path}\n")

print(f"\nAll results saved to {output_dir}/")
