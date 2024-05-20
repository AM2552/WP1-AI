import os
import shutil
from tqdm import tqdm


############################################
source_folder = '/Users/Xandi/Desktop/Datasets/PetImages/dog'
target_folder = '/Users/Xandi/Desktop/Datasets/bcd/dog'
############################################
          
def copy_images_multifolder():
    for folder in tqdm(os.listdir(source_folder), desc="Processing folders"):
        folder_path = os.path.join(source_folder, folder)
        if os.path.isdir(folder_path):
            for image in tqdm(os.listdir(folder_path), desc="Processing images"):
                image_path = os.path.join(folder_path, image)
                if image.endswith(('.jpg', '.jpeg', '.png')):
                    destination = os.path.join(target_folder)
                    if not os.path.exists(destination):
                        os.mkdir(destination)
                    shutil.copy(image_path, destination)

def copy_images():
    for image in tqdm(os.listdir(source_folder), desc="Processing images"):
        image_path = os.path.join(source_folder, image)
        if image.endswith(('.jpg', '.jpeg', '.png')):
            destination = os.path.join(target_folder)
            if not os.path.exists(destination):
                os.mkdir(destination)
            shutil.copy(image_path, destination)

if __name__ == '__main__':
    copy_images()