from PIL import Image
import os
from tqdm import tqdm

# datasetname/
#     training/
#         cat/
#         dog/
#     validation/
#         cat/
#         dog/

dataset_path = 'datasets/birds/'

def pad_image(image_path, output_path):
    try:
        img = Image.open(image_path)
        width, height = img.size
        size = max(width, height)
        new_img = Image.new("RGB", (size, size))
        new_img.paste(img, ((size - width) // 2, (size - height) // 2))
        new_img.save(output_path)
    except OSError:
        print(f"Skipping file due to OSError: {output_path}")

for datatype in tqdm(os.listdir(dataset_path), desc="Processing datatypes"):
    for label in tqdm(os.listdir(os.path.join(dataset_path, datatype)), desc="Processing classes"):
        for filename in tqdm(os.listdir(os.path.join(dataset_path, datatype, label)), desc="Processing images"):
            if filename.endswith(('.png', '.jpg', '.jpeg')): 
                image_path = os.path.join(dataset_path, datatype, label, filename)
                pad_image(image_path, image_path)

#for folder in tqdm(os.listdir(dataset_path), desc="Processing classes"):
#    for filename in tqdm(os.listdir(os.path.join(dataset_path, folder)), desc="Processing images"):
#        if filename.endswith(('.png', '.jpg', '.jpeg')): 
#            image_path = os.path.join(dataset_path, folder, filename)
#            pad_image(image_path, image_path)