from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

model = load_model('cats_vs_dogs_model_Preset1.h5')

folder_path = './cats_dogs/test_images'
#folder_path = './datasets/cat_vs_dog/test'
accuracy_counter = 0
dog_accuracy = 0
cat_accuracy = 0

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
        dog_counter = len(os.listdir(os.path.join(folder_path, 'dog')))
        cat_counter = len(os.listdir(os.path.join(folder_path, 'cat')))
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, folder, filename)
            img = pad_image(img_path)
            #img = Image.open(img_path)
            img = img.resize((256, 256))
            
            #plt.imshow(img)
            #plt.title(f"Resized Image: {filename}")
            #plt.show()

            img_array = image.img_to_array(img)
            img_array /= 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)
            class_name = "dog" if prediction[0][0] > 0.5 else "cat"
            probability = prediction[0][0] * 100 if class_name == "dog" else (1 - prediction[0][0]) * 100
            
            # get prob per class
            if class_name == 'dog' and class_name in folder:
                dog_accuracy += 1
            elif class_name == 'cat' and class_name in folder:
                cat_accuracy += 1

            #print(f"{filename}: The image is a {class_name} with probability {probability:.2f}%")

            #with open('accuracy.txt', 'w') as f:
            #    f.write(f"{filename}: The image is a {class_name} with probability {probability:.2f}%")

total_images = len(os.listdir(folder_path))
dog_accuracy = dog_accuracy / dog_counter * 100
cat_accuracy = cat_accuracy / cat_counter * 100
val_accuracy = (dog_accuracy + cat_accuracy) / total_images
with open('test_report.txt', 'w') as f:
    f.write(f"Dog accuracy: {dog_accuracy:.2f}%\n")
    f.write(f"Cat accuracy: {cat_accuracy:.2f}%\n")
    f.write(f"Testdata accuracy: {val_accuracy:.2f}%\n")
print(f"Testdata accuracy: {val_accuracy:.2f}%")
            
