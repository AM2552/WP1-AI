import os
import json
import cv2  # pip install opencv-python
from ultralytics import YOLO  # pip install ultralytics


# Update these paths as needed
ROOT_DIR = "yolo/amphibia"
JSON_ANNOT_PATH = "amphibians/amphibia_annotations.json"  # The JSON with bounding boxes for all images
DATA_CONFIG_PATH = os.path.join(ROOT_DIR, "amphibia.yaml")

# If you have 16 species, list them here in the exact order that matches your class indices.
# If your JSON uses class labels 0..15 or 1..16, be consistent.
CLASS_NAMES = [
    'feuersalamander',
    'alpensalamander',
    'bergmolch',
    'kammmolch',
    'teichmolch',
    'rotbauchunke',
    'gelbbauchunke',
    'knoblauchkröte',
    'erdkröte',
    'kreuzkröte',
    'wechselkröte',
    'laubfrosch',
    'moorfrosch',
    'springfrosch',
    'grasfrosch',
    'wasserfrosch'
]

########################################################
# 1) READ JSON ANNOTATIONS AND STORE THEM FOR FAST LOOKUP
########################################################
with open(JSON_ANNOT_PATH, "r") as f:
    annotations = json.load(f)

# We’ll create a dict mapping each image filename to a list of (class_label, bbox).
# For instance:
#   ann_dict["004908b9.jpg"] = [
#       (2, {"x_min":..., "y_min":..., "x_max":..., "y_max":...}),
#       ...
#   ]
ann_dict = {}

for ann in annotations:
    image_path = ann["image_path"]   # e.g. "004908b9-ab3e-4223-9a84-190e52e14687.jpg"
    bboxes = ann["bboxes"]          # list of dicts
    if image_path not in ann_dict:
        ann_dict[image_path] = []
    for b in bboxes:
        class_label = b["class_label"]
        bbox_coords = b["bbox"]
        # If your JSON is 1-based classes, do: class_label -= 1
        ann_dict[image_path].append((class_label-1, bbox_coords))

########################################################
# 2) FUNCTION TO CONVERT ABSOLUTE BBOX => YOLO FORMAT
########################################################
def convert_bbox_to_yolo(img_w, img_h, bbox):
    """
    bbox: dict with x_min, y_min, x_max, y_max
    Returns (x_center_norm, y_center_norm, w_norm, h_norm)
    """
    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]

    w = x_max - x_min
    h = y_max - y_min
    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0

    # Normalize
    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h
    return x_center, y_center, w, h

########################################################
# 3) WALK THE EXISTING FOLDER STRUCTURE & CREATE LABELS
########################################################
splits = ["training", "validation", "test"]

for split in splits:
    split_dir = os.path.join(ROOT_DIR, split)
    
    # For each species subfolder in this split
    species_subfolders = [
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ]
    
    for species_folder in species_subfolders:
        species_path = os.path.join(split_dir, species_folder)
        
        # For each image in this species folder
        for filename in os.listdir(species_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            # Check if we have annotations for this file
            if filename not in ann_dict:
                # No bounding boxes in the JSON => either skip or create an empty .txt
                # If you want to create an empty txt for “no objects”, do:
                # open(os.path.join(species_path, filename.replace('.jpg', '.txt')), 'w').close()
                continue
            
            # Read image to get width/height
            img_path = os.path.join(species_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: cannot read {img_path}. Skipping.")
                continue
            
            h, w, _ = img.shape
            
            # Create the label file with the same base name
            label_file = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(species_path, label_file)
            
            # Write each bounding box in YOLO format
            with open(label_path, "w") as lf:
                for (class_label, bbox_dict) in ann_dict[filename]:
                    # If your JSON class_label is 1-based, subtract 1:
                    # class_label = class_label - 1
                    xC, yC, boxW, boxH = convert_bbox_to_yolo(w, h, bbox_dict)
                    lf.write(f"{class_label} {xC} {yC} {boxW} {boxH}\n")

print("Finished writing YOLO .txt labels alongside images.")

########################################################
# 4) CREATE A YOLO DATA CONFIG FILE (amphibia.yaml)
########################################################
# We can point YOLO to exactly where your splits are, telling it:
#   train: path/to/datasets/amphibia/training
#   val:   path/to/datasets/amphibia/validation
#   test:  path/to/datasets/amphibia/test


with open(DATA_CONFIG_PATH, "w") as f:
    f.write("names:\n")
    for c in CLASS_NAMES:
        f.write(f"  - {c}\n")
    
    # If your classes are 16 species with the same indices used in your JSON, ensure
    # the order here matches your numeric labels after any offset.
    
    f.write(f"train: {os.path.join('C:/Users/Xandi/Documents/Repository/WP1-AI/yolo/amphibia', 'training')}\n")
    f.write(f"val: {os.path.join('C:/Users/Xandi/Documents/Repository/WP1-AI/yolo/amphibia', 'validation')}\n")
    f.write(f"test: {os.path.join('C:/Users/Xandi/Documents/Repository/WP1-AI/yolo/amphibia', 'test')}\n")

print(f"Wrote YOLO data config to {DATA_CONFIG_PATH}.")

