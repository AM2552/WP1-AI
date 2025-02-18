import os
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm

# Label mapping
label_mapping = {
    'feuersalamander': 1,
    'alpensalamander': 2,
    'bergmolch': 3,
    'kammmolch': 4,
    'teichmolch': 5,
    'rotbauchunke': 6,
    'gelbbauchunke': 7,
    'knoblauchkröte': 8,
    'erdkröte': 9,
    'kreuzkröte': 10,
    'wechselkröte': 11,
    'laubfrosch': 12,
    'moorfrosch': 13,
    'springfrosch': 14,
    'grasfrosch': 15,
    'wasserfrosch': 16
}

def parse_xml(xml_file, image_counter):
    """ Parses an XML file and extracts images with bounding boxes. """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []

    for image in root.findall('image'):
        image_name = image.get('name')
        bboxes = []
        
        for box in image.findall('box'):
            label = box.get('label')
            x_min = float(box.get('xtl'))
            y_min = float(box.get('ytl'))
            x_max = float(box.get('xbr'))
            y_max = float(box.get('ybr'))

            bbox = {
                "class_label": label_mapping.get(label, -1),  # Assign -1 if label is unknown
                "bbox": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                }
            }
            bboxes.append(bbox)

        # Only include images that have bounding boxes
        if bboxes:
            annotation = {
                "image_number": image_counter,  # Unique and incrementing image number
                "image_path": image_name,
                "bboxes": bboxes
            }
            annotations.append(annotation)
            image_counter += 1  # Increment the global counter
    
    return annotations, image_counter

def process_xml_folder(input_folder, output_json):
    """ Processes all XML files in the input folder and saves a single JSON file. """
    xml_files = [f for f in os.listdir(input_folder) if f.endswith('.xml')]
    all_annotations = []
    image_counter = 1  # Start numbering from 1

    for xml_file in tqdm(xml_files, desc="Processing XML files"):
        xml_path = os.path.join(input_folder, xml_file)
        annotations, image_counter = parse_xml(xml_path, image_counter)
        
        if annotations:
            all_annotations.extend(annotations)
        else:
            print(f"Skipping {xml_file} (no bounding boxes found).")

    # Save all annotations in a single JSON file
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=4)

    print(f"Processing complete. JSON saved as '{output_json}' with {len(all_annotations)} annotated images.")

if __name__ == "__main__":
    input_folder = "merged_annotations"
    output_json = "merged_annotations/combined_annotations.json"

    process_xml_folder(input_folder, output_json)
