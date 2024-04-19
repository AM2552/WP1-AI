import os
import shutil
from tqdm import tqdm

"""
source_folder should look like this:

images/
  class1/
    img1.jpg
    img2.jpg
    ...
  class2/
    img1.jpg
    img2.jpg
    ...

split_dataset will look like this:

dataset/
  training/
    class1/
      img1.jpg
      ...
    class2/
      img1.jpg
      ...
  validation/
    class1/
      img1.jpg
      ...
    class2/
      img1.jpg
      ...
  test/
    class1/
      img1.jpg
      ...
    class2/
      img1.jpg
      ...
"""

############################################
source_folder = '/Users/Xandi/Desktop/Datasets/PetImages'
new_dataset_path = './datasets/cat_vs_dog'
############################################
          
def split_dataset():
    classes = os.listdir(source_folder)
    training_path = os.path.join(new_dataset_path, 'training')
    validation_path = os.path.join(new_dataset_path, 'validation')
    test_path = os.path.join(new_dataset_path, 'test')
    
    for path in [new_dataset_path, training_path, validation_path, test_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    
    for element in tqdm(classes, desc="Processing classes"):
        class_folder = os.path.join(source_folder, element)
        images = os.listdir(class_folder)
        
        # split ratio 70/15/15
        total_images = len(images)
        train_split_idx = int(total_images * 0.7)
        validation_split_idx = train_split_idx + int(total_images * 0.15)
        
        train_images = images[:train_split_idx]
        validation_images = images[train_split_idx:validation_split_idx]
        test_images = images[validation_split_idx:]
        
        class_paths = {
            'training': os.path.join(training_path, element),
            'validation': os.path.join(validation_path, element),
            'test': os.path.join(test_path, element)
        }
        
        for path in class_paths.values():
            if not os.path.exists(path):
                os.mkdir(path)
        
        for image_set, path in tqdm(zip([train_images, validation_images, test_images], class_paths.values()), desc="Processing images"):
            for image_name in image_set:
                source = os.path.join(class_folder, image_name)
                destination = os.path.join(path, image_name)
                shutil.copyfile(source, destination)
        
if __name__ == '__main__':
    split_dataset()