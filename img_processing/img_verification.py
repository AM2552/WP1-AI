from PIL import Image
import os
from tqdm import tqdm

image_dir = '/Users/Xandi/Desktop/Datasets/bcd/bird'
problematic_files = []

for filename in tqdm(os.listdir(image_dir), desc="Processing images"):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        try:
            with Image.open(os.path.join(image_dir, filename)) as img:
                img.verify()
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)
            problematic_files.append(filename)


print("Problematic files:", problematic_files)
# for file in problematic_files:
#     os.remove(os.path.join(image_dir, file))