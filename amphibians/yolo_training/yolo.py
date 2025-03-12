import os
from ultralytics import YOLO


# Update these paths as needed
ROOT_DIR = "yolo/amphibia"
JSON_ANNOT_PATH = "amphibians/amphibia_annotations.json"  # The JSON with bounding boxes for all images
DATA_CONFIG_PATH = os.path.join(ROOT_DIR, "amphibia.yaml")

########################################################
# 5) TRAIN A YOLOv8 MODEL
########################################################
# Example: train a YOLOv8n or YOLOv8s model for ~100 epochs

def main():
    model = YOLO("yolo11x.pt")  # or yolov8s.pt, etc.
    results = model.train(
        data=DATA_CONFIG_PATH,  # "datasets/amphibia/amphibia.yaml"
        epochs=100,
        imgsz=640,
        project="amphibians/runs/train",
        name="yolo",
        pretrained=True,
        batch=4,
    )

    print("Training complete. Check runs/train/amphibia_yolo for results.")

if __name__ == "__main__":
    main()
