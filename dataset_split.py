import os

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

############################################
source_folder = '/Users/z004wr2n/Downloads/archive/PetImages'
new_dataset_path = '/Users/z004wr2n/Downloads/archive/datasets/cat&dog'
############################################
          
def split_dataset(split_ratio=0.8):
    classes = os.listdir(source_folder)
    training_path = new_dataset_path +'/training'
    validation_path = new_dataset_path +'/validation'
    
    if not os.path.exists(new_dataset_path):
        os.mkdir(new_dataset_path)
    if not os.path.exists(training_path):
        os.mkdir(training_path)
    if not os.path.exists(validation_path):
        os.mkdir(validation_path)
    
    for element in classes:
        class_training_path = training_path + '/' + element
        if not os.path.exists(class_training_path):
            os.mkdir(class_training_path)
        class_validation_path = validation_path + '/' + element
        if not os.path.exists(class_validation_path):
            os.mkdir(class_validation_path)
        
        class_folder = source_folder + '/' + element
        dataset = os.listdir(class_folder)
        train_size = int(len(dataset) * split_ratio)
        train_set = dataset[:train_size]
        test_set = dataset[train_size:]
        for img in train_set:
            os.rename(os.path.join(class_folder, img), os.path.join(class_training_path, img))
        for img in test_set:
            os.rename(os.path.join(class_folder, img), os.path.join(class_validation_path, img))
        
if __name__ == '__main__':
    split_dataset()