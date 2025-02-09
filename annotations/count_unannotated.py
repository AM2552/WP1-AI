import xml.etree.ElementTree as ET

def count_unannotated_images(xml_file_path):
    # Load and parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Count images without annotations
    total_images = 0
    unannotated_images = 0

    for image in root.findall("image"):
        total_images += 1
        if not image.findall("box"):  # If no 'box' elements exist inside 'image'
            unannotated_images += 1

    print(f"Total images: {total_images}")
    print(f"Unannotated images: {unannotated_images}")

# Replace 'your_file.xml' with your actual XML file path
xml_file_path = "annotations/feuersalamander.xml"
count_unannotated_images(xml_file_path)
