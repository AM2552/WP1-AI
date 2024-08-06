import pandas as pd
import json

# Load the Excel file
file_path = 'birds_OD/birds_annotations.xlsx'
data = pd.read_excel(file_path)

# Process the data
annotations = []

for index, row in data.iterrows():
    bbox_values = row['bboxes'].split()
    bbox = {
        "x_min": float(bbox_values[0]),
        "y_min": float(bbox_values[1]),
        "x_max": float(bbox_values[2]),
        "y_max": float(bbox_values[3])
    }
    annotation = {
        "class_label": row["class_labels"],
        "image_number": row["image_number"],
        "image_path": row["image_path"],
        "bbox": bbox
    }
    annotations.append(annotation)

# Save to JSON
output_path = 'birds_OD/birds_annotations.json'
with open(output_path, 'w') as json_file:
    json.dump(annotations, json_file, indent=4)

print(f"Annotations saved to {output_path}")
