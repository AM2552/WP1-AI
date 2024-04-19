from PIL import Image
import os

image_dir = '/Users/Xandi/Desktop/Datasets/PetImages/Dog/'
problematic_files = []

for filename in os.listdir(image_dir):
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