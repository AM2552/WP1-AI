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
model = load_model('bird_cat_dog_model_916.h5')

# Set the path to the folder containing the images
folder_path = 'datasets/bird_cat_dog/test'
output_path = './bird_cat_dog/test_images/grad_cam'
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

def get_img_array(img_path, size):
    img = load_img(img_path, target_size=size)
    array = img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=1.0):
    img = load_img(img_path)
    img = img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = plt.colormaps.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)

for folder in tqdm(os.listdir(folder_path), desc="Processing classes"):
    for filename in tqdm(os.listdir(os.path.join(folder_path, folder)), desc="Processing images"):
        bird_counter = len(os.listdir(os.path.join(folder_path, 'bird')))
        cat_counter = len(os.listdir(os.path.join(folder_path, 'cat')))
        dog_counter = len(os.listdir(os.path.join(folder_path, 'dog')))
        
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

            last_conv_layer_name = "conv2d_5"  # Update with the correct name of your model's last convolutional layer
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            save_and_display_gradcam(img_path, heatmap, cam_path=os.path.join(output_path, folder, f"grad_cam_{filename}"))
            if class_name != folder:
                with open("image_details.txt", "a") as f:
                    f.write(f"Image: {filename} - ")
                    f.write(f"Class: {class_name} - ")
                    f.write(f"Probability: {probability}\n")

total_images = len(os.listdir(folder_path))
dog_accuracy = dog_accuracy / dog_counter * 100
cat_accuracy = cat_accuracy / cat_counter * 100
bird_accuracy = bird_accuracy / bird_counter * 100
val_accuracy = (bird_accuracy + cat_accuracy + dog_accuracy) / total_images

with open('test_report.txt', 'w') as f:
    f.write(f"Bird accuracy: {bird_accuracy:.2f}%\n")
    f.write(f"Cat accuracy: {cat_accuracy:.2f}%\n")
    f.write(f"Dog accuracy: {dog_accuracy:.2f}%\n")
    f.write(f"Testdata accuracy: {val_accuracy:.2f}%\n")

print(f"Testdata accuracy: {val_accuracy:.2f}%")
