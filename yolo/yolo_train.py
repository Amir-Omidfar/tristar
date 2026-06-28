"""
YOLO11 Fast Training Pipeline for White Bracket Defect Detection
"""
import os
from ultralytics import YOLO

def train_white_bracket_yolo():
    # 1. Initialize YOLO11 Nano Model (Pre-trained on COCO features)
    # The 'n' model is incredibly lightweight, fast to train, and perfect for quick proof-of-concept testing
    model = YOLO("yolo11n.pt") 
    
    # 2. Kick off the training loop
    print("🚀 Initializing YOLO11 Training Loop on White Bracket Subclass...")
    results = model.train(
        data="/Users/ApplePro/Desktop/gitProjects/tristar/yolo/dataset.yaml",      # Path to your configuration file
        epochs=50,                # 50 epochs is enough for a fast verification test
        imgsz=640,                # YOLO natively performs best at 640x640 resolution
        batch=16,                 # Safe batch size to avoid GPU memory fragmentation
        workers=2,                # Multi-threaded data loading workers
        name="yolo11_white_bracket", # Output folder name inside runs/detect/
        plots=True,                
        device="cpu"
    )
    
    print("\n✅ Training complete! Best weights saved to: runs/detect/yolo11_white_bracket/weights/best.pt")

if __name__ == "__main__":
    train_white_bracket_yolo()