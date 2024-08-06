import xml.etree.ElementTree as ET
import json

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

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []

    for image in root.findall('image'):
        image_id = int(image.get('id'))
        image_name = image.get('name')
        bboxes = []
        
        for box in image.findall('box'):
            label = box.get('label')
            x_min = float(box.get('xtl'))
            y_min = float(box.get('ytl'))
            x_max = float(box.get('xbr'))
            y_max = float(box.get('ybr'))

            bbox = {
                "class_label": label_mapping[label],
                "bbox": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                }
            }
            bboxes.append(bbox)

        annotation = {
            "image_number": image_id + 1,
            "image_path": image_name,
            "bboxes": bboxes
        }
        annotations.append(annotation)
    
    return annotations

def write_json(annotations, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    xml_file = 'amphibians/annotations.xml'
    output_file = 'amphibians/amphibia_annotations.json'
    
    annotations = parse_xml(xml_file)
    write_json(annotations, output_file)