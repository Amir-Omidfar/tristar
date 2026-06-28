import os
from glob import glob
from ultralytics import YOLO

yolo_model = YOLO("/Users/ApplePro/Desktop/gitProjects/tristar/runs/detect/yolo11_white_bracket-4/weights/best.pt")
def test_yolo(image_path):
    results = yolo_model(image_path)
    boxes = results[0].boxes

    if len(boxes) > 0:
        verdict = "REJECT (Defective Part)"
        details = f"Detected {len(boxes)} defects using YOLO11 detection frame."
        print(f"{image_path},   {verdict},   {details}")
    else:
        verdict = "PASS (Good Part)"
        print(f"{image_path},   {verdict}")
    return verdict,details

directory = "/Users/ApplePro/Desktop/gitProjects/tristar/yolo/white_bracket_yolo_data/test_images/"
for file_name in os.listdir(directory):
    test_yolo(f"{directory}{file_name}")

