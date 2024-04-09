from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the model
model = load_model('cats_vs_dogs_model.h5')

# Specify the path to your folder containing the images
folder_path = './test_images/'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add/check your image file extensions here
        img_path = os.path.join(folder_path, filename)
        img = image.load_img(img_path, target_size=(224, 224))  # Load and resize the image

        ## Display the resized image
        #plt.imshow(img)
        #plt.title(f"Resized Image: {filename}")
        #plt.show()

        img_array = image.img_to_array(img)  # Convert the image to a numpy array
        img_array /= 255.0  # Scale the image pixels
        img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

        # Make a prediction
        prediction = model.predict(img_array)

        # Print the filename, predicted class, and probability
        class_name = "dog" if prediction[0][0] > 0.5 else "cat"
        probability = prediction[0][0] * 100 if class_name == "dog" else (1 - prediction[0][0]) * 100
        print(f"{filename}: The image is a {class_name} with probability {probability:.2f}%")
