from tensorflow import keras
from keras.models import load_model, Model
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from keras.utils import load_img, img_to_array

# Load the pre-trained model
model = load_model('bird_cat_dog_model_Preset3_continued.h5')

# Set the path to the folder containing the images
folder_path = 'datasets/bird_cat_dog/test'
class_names = ['bird', 'cat', 'dog']

accuracy_counter = 0
bird_accuracy = 0
cat_accuracy = 0
dog_accuracy = 0

def pad_image(image_path):
    try:
        img = Image.open(image_path)
        width, height = img.size
        size = max(width, height)
        new_img = Image.new("RGB", (size, size))
        new_img.paste(img, ((size - width) // 2, (size - height) // 2))
        return new_img
    except OSError:
        print(f"Skipping file due to OSError: {image_path}")

for folder in tqdm(os.listdir(folder_path), desc="Processing classes"):
    for filename in tqdm(os.listdir(os.path.join(folder_path, folder)), desc="Processing images"):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, folder, filename)
            img = pad_image(img_path)
            img = img.resize((256, 256))

            img_array = img_to_array(img)
            img_array /= 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)
            class_name = class_names[np.argmax(prediction[0])]
            probability = np.max(prediction[0])
            
            if class_name == folder:
                accuracy_counter += 1
                if folder == 'bird':
                    bird_accuracy += 1
                elif folder == 'cat':
                    cat_accuracy += 1
                elif folder == 'dog':
                    dog_accuracy += 1

bird_counter = len(os.listdir(os.path.join(folder_path, 'bird')))
cat_counter = len(os.listdir(os.path.join(folder_path, 'cat')))
dog_counter = len(os.listdir(os.path.join(folder_path, 'dog')))

total_images = len(os.listdir(folder_path))
dog_accuracy = dog_accuracy / dog_counter * 100
cat_accuracy = cat_accuracy / cat_counter * 100
bird_accuracy = bird_accuracy / bird_counter * 100
val_accuracy = (dog_accuracy + bird_accuracy + cat_accuracy) / total_images

with open('test_report.txt', 'w') as f:
    f.write(f"Bird accuracy: {bird_accuracy:.2f}%\n")
    f.write(f"Cat accuracy: {cat_accuracy:.2f}%\n")
    f.write(f"Dog accuracy: {dog_accuracy:.2f}%\n")
    f.write(f"Testdata accuracy: {val_accuracy:.2f}%\n")

print(f"Testdata accuracy: {val_accuracy:.2f}%")