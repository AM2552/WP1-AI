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
"""
# images folder contains 200 classes of birds

############################################
source_folder = '/Users/Xandi/Desktop/Datasets/birds/CUB_200_2011/images'
new_dataset_path = './datasets/birds'
############################################
          
def split_dataset():
    classes = os.listdir(source_folder)
    training_path = os.path.join(new_dataset_path, 'training')
    validation_path = os.path.join(new_dataset_path, 'validation')
    
    for path in [new_dataset_path, training_path, validation_path]:
        os.makedirs(path, exist_ok=True)
    
    for class_name in tqdm(classes, desc="Processing classes"):
        class_folder = os.path.join(source_folder, class_name)
        images = [img for img in os.listdir(class_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # split ratio 70/30
        total_images = len(images)
        train_split = int(total_images * 0.7)
        
        train_images = images[:train_split]
        validation_images = images[train_split:]
        
        class_paths = {
            'training': os.path.join(training_path, class_name),
            'validation': os.path.join(validation_path, class_name)
        }
        
        for datatype, path in class_paths.items():
            os.makedirs(path, exist_ok=True)
            image_set = train_images if datatype == 'training' else validation_images
            for image_name in tqdm(image_set, desc="Processing images"):
                source = os.path.join(class_folder, image_name)
                destination = os.path.join(path, image_name)
                try:
                    shutil.copyfile(source, destination)
                except IOError as e:
                    print(f"Error copying {image_name}: {e}")
        
if __name__ == '__main__':
    split_dataset()