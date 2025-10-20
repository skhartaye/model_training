"""
Training script for Philippine License Plate Detection
"""

import os
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_yolov5():
    """Clone YOLOv5 repository if not present"""
    yolov5_dir = Path('yolov5')
    if not yolov5_dir.exists():
        logger.info("Cloning YOLOv5 repository...")
        os.system('git clone https://github.com/ultralytics/yolov5.git')
        os.system('pip install -r yolov5/requirements.txt')
    return yolov5_dir


def prepare_dataset():
    """Prepare the dataset configuration"""
    dataset_dir = Path('Philippine_License_Plate.v1i.yolov5-obb')
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return None
    
    # Check if labels need conversion
    train_labeltxt = dataset_dir / 'train' / 'labelTxt'
    if train_labeltxt.exists():
        logger.info("Converting OBB labels to YOLO format...")
        convert_obb_to_yolo(dataset_dir)
    
    # Update data.yaml
    data_yaml_path = dataset_dir / 'data.yaml'
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['path'] = str(dataset_dir.absolute())
    config['train'] = 'train/images'
    config['val'] = 'valid/images'
    config['test'] = 'test/images'
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Dataset ready: {config['nc']} classes - {config['names']}")
    return data_yaml_path


def convert_obb_to_yolo(dataset_dir):
    """Convert OBB format to standard YOLO format"""
    import cv2
    
    for split in ['train', 'valid', 'test']:
        labeltxt_dir = dataset_dir / split / 'labelTxt'
        labels_dir = dataset_dir / split / 'labels'
        images_dir = dataset_dir / split / 'images'
        
        if not labeltxt_dir.exists():
            continue
        
        labels_dir.mkdir(exist_ok=True)
        label_files = list(labeltxt_dir.glob('*.txt'))
        logger.info(f"Converting {len(label_files)} {split} labels...")
        
        for label_file in label_files:
            try:
                # Find corresponding image to get dimensions
                img_name = label_file.stem + '.jpg'
                img_path = images_dir / img_name
                
                if not img_path.exists():
                    img_name = label_file.stem + '.png'
                    img_path = images_dir / img_name
                
                if not img_path.exists():
                    logger.warning(f"Image not found for {label_file.name}")
                    continue
                
                # Get image dimensions
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_height, img_width = img.shape[:2]
                
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                yolo_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        coords = [float(x) for x in parts[:8]]
                        class_name = parts[8]
                        
                        class_map = {'Invalid-Plate': 0, 'License-Plate': 1, 'Plate-Content': 2}
                        class_id = class_map.get(class_name, 1)
                        
                        x_coords = coords[0::2]
                        y_coords = coords[1::2]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        # Normalize coordinates
                        x_center = ((x_min + x_max) / 2) / img_width
                        y_center = ((y_min + y_max) / 2) / img_height
                        width = (x_max - x_min) / img_width
                        height = (y_max - y_min) / img_height
                        
                        # Ensure values are in valid range
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")
                
                if yolo_lines:
                    with open(labels_dir / label_file.name, 'w') as f:
                        f.writelines(yolo_lines)
            except Exception as e:
                logger.warning(f"Error converting {label_file.name}: {e}")


def train_model(data_yaml, epochs=100, batch_size=16, img_size=640):
    """Train YOLOv5n model"""
    cmd = (
        f"python yolov5/train.py "
        f"--img {img_size} --batch {batch_size} --epochs {epochs} "
        f"--data {data_yaml} --weights yolov5n.pt "
        f"--project runs/train --name philippine_plates --cache"
    )
    logger.info(f"Training command: {cmd}")
    os.system(cmd)


def main():
    print("="*60)
    print("Philippine License Plate Detection - Training")
    print("="*60)
    
    setup_yolov5()
    data_yaml = prepare_dataset()
    
    if data_yaml is None:
        return
    
    epochs = int(input("Epochs (default 100): ") or "100")
    batch_size = int(input("Batch size (default 16): ") or "16")
    
    train_model(data_yaml, epochs, batch_size)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("Best model: runs/train/philippine_plates/weights/best.pt")
    print("="*60)


if __name__ == "__main__":
    main()
