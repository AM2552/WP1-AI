from PIL import Image
import os

image_dir = '/Users/Xandi/Desktop/Datasets/PetImages/Dog/'  # Update this to the path of your images
problematic_files = []

for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):  # Add any other file extensions you expect
        try:
            with Image.open(os.path.join(image_dir, filename)) as img:
                img.verify()  # Verify that it is, in fact, an image
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)  # Print out the names of corrupt files
            problematic_files.append(filename)

# Optionally, print or remove the problematic files
print("Problematic files:", problematic_files)
# for file in problematic_files:
#     os.remove(os.path.join(image_dir, file))  # Be cautious with this