from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras import backend
from keras.callbacks import Callback
from keras.applications import EfficientNetB1
from dataset_generation_bcd import train_generator, validation_generator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array, array_to_img
from PIL import Image
import os

if not os.path.exists('bird_cat_dog/test_images/grad_cam/bird'):
    os.makedirs('bird_cat_dog/test_images/grad_cam/bird')
if not os.path.exists('bird_cat_dog/test_images/grad_cam/cat'):
    os.makedirs('bird_cat_dog/test_images/grad_cam/cat')
if not os.path.exists('bird_cat_dog/test_images/grad_cam/dog'):
    os.makedirs('bird_cat_dog/test_images/grad_cam/dog')

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, epsilon=1e-5):
    grad_model = keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    grads += epsilon  # Add a small value to gradients to avoid zero gradients

    if grads is None:
        print("Warning: Gradients are None, skipping this step.")
        return np.zeros_like(last_conv_layer_output[0, :, :, 0]), preds

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = np.nan_to_num(heatmap)  # Replace NaN with zero and infinity with large finite numbers
    return heatmap, preds

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.95):
    img = load_img(img_path)
    img = img_to_array(img)

    # Ensure the heatmap has the correct shape
    heatmap = np.uint8(255 * heatmap)
    if len(heatmap.shape) == 2:
        heatmap = np.expand_dims(heatmap, axis=-1)

    # Resize heatmap to the image size
    heatmap = array_to_img(heatmap)
    heatmap = heatmap.resize((img.shape[1], img.shape[0]))
    heatmap = img_to_array(heatmap)

    # Convert the heatmap to RGB
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap[:, :, 0].astype(int)]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)

class GradCAMCallback(Callback):
    def __init__(self, model, bird_img_path, cat_img_path, dog_img_path, last_conv_layer_name):
        super(GradCAMCallback, self).__init__()
        self.model = model
        self.bird_img_path = bird_img_path
        self.cat_img_path = cat_img_path
        self.dog_img_path = dog_img_path
        self.last_conv_layer_name = last_conv_layer_name
        self.class_names = {'bird': 0, 'cat': 1, 'dog': 2}

    def on_epoch_end(self, epoch, logs=None):
        bird_img_array, original_bird_width, original_bird_height = get_img_array(self.bird_img_path, (256, 256))
        cat_img_array, original_cat_width, original_cat_height = get_img_array(self.cat_img_path, (256, 256))
        dog_img_array, original_dog_width, original_dog_height = get_img_array(self.dog_img_path, (256, 256))

        if bird_img_array is not None:
            bird_heatmap, bird_preds = make_gradcam_heatmap(bird_img_array, self.model, self.last_conv_layer_name, pred_index=self.class_names['bird'])
            save_and_display_gradcam(self.bird_img_path, bird_heatmap, cam_path=f'bird_cat_dog/test_images/grad_cam/bird/grad_cam_bird_epoch_{epoch + 1}.jpg')
            bird_pred_class = "bird" if tf.argmax(bird_preds[0]) == 0 else "cat" if tf.argmax(bird_preds[0]) == 1 else "dog"
            bird_confidence = tf.reduce_max(bird_preds[0])
            with open('training_log.txt', 'a') as f:
                f.write(f'Epoch {epoch + 1}: Bird image - Predicted class: {bird_pred_class}, Confidence: {bird_confidence:.4f}\n')

        if cat_img_array is not None:
            cat_heatmap, cat_preds = make_gradcam_heatmap(cat_img_array, self.model, self.last_conv_layer_name, pred_index=self.class_names['cat'])
            save_and_display_gradcam(self.cat_img_path, cat_heatmap, cam_path=f'bird_cat_dog/test_images/grad_cam/cat/grad_cam_cat_epoch_{epoch + 1}.jpg')
            cat_pred_class = "bird" if tf.argmax(cat_preds[0]) == 0 else "cat" if tf.argmax(cat_preds[0]) == 1 else "dog"
            cat_confidence = tf.reduce_max(cat_preds[0])
            with open('training_log.txt', 'a') as f:
                f.write(f'Epoch {epoch + 1}: Cat image - Predicted class: {cat_pred_class}, Confidence: {cat_confidence:.4f}\n')

        if dog_img_array is not None:
            dog_heatmap, dog_preds = make_gradcam_heatmap(dog_img_array, self.model, self.last_conv_layer_name, pred_index=self.class_names['dog'])
            save_and_display_gradcam(self.dog_img_path, dog_heatmap, cam_path=f'bird_cat_dog/test_images/grad_cam/dog/grad_cam_dog_epoch_{epoch + 1}.jpg')
            dog_pred_class = "bird" if tf.argmax(dog_preds[0]) == 0 else "cat" if tf.argmax(dog_preds[0]) == 1 else "dog"
            dog_confidence = tf.reduce_max(dog_preds[0])
            with open('training_log.txt', 'a') as f:
                f.write(f'Epoch {epoch + 1}: Dog image - Predicted class: {dog_pred_class}, Confidence: {dog_confidence:.4f}\n')

def get_img_array(img_path, size):
    img = Image.open(img_path)  # No padding or resizing required
    original_width, original_height = img.size
    img = img.resize(size)
    array = img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array, original_width, original_height

def train_model(dense_layers, learning_rate, epochs, dropout=bool, bird_img_path=str(), cat_img_path=str(), dog_img_path=str(), last_conv_layer_name=str(), preset_name=str()):
    base_model = EfficientNetB1(include_top=False, weights=None, input_shape=(256, 256, 3))
    base_model.trainable = True
    
    model = Sequential()
    model.add(base_model)
    
    model.add(GlobalAveragePooling2D())
    neurons = 512
    for _ in range(dense_layers):
        model.add(Dense(neurons))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        neurons /= 2
    if dropout:
        model.add(Dropout(0.3))
    model.add(Dense(neurons))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(3, activation='softmax'))

    model.summary()
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    grad_cam_callback = GradCAMCallback(model, bird_img_path, cat_img_path, dog_img_path, last_conv_layer_name)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[grad_cam_callback]
    )
    
    accuracy_history = history.history['val_accuracy']
    #model.save(f'bird_cat_dog_model_{preset_name}_eff.h5')
    
    return accuracy_history

results = []

parameter_presets = {
    'Preset1': (3, 0.0001, 50, True, 'bird_cat_dog/test_images/bird/bird1.jpg', 'bird_cat_dog/test_images/cat/cat2.jpg', 'bird_cat_dog/test_images/dog/dog2.jpg', 'efficientnetb1')
    #'Preset2': (6, 4, 0.0003, 50, True, './cats_dogs/test_images/cat/cat1.jpg', './cats_dogs/test_images/dog/dog1.jpg', 'conv2d_5'),
    #'Preset3': (6, 3, 0.0001, 70, True, './cats_dogs/test_images/cat/cat1.jpg', './cats_dogs/test_images/dog/dog1.jpg', 'conv2d_5'),
    #'Preset4': (6, 3, 0.0003, 50, True, './cats_dogs/test_images/cat/cat1.jpg', './cats_dogs/test_images/dog/dog1.jpg', 'conv2d_5')
}

for preset_name, parameters in parameter_presets.items():
    backend.clear_session()
    accuracy_history = train_model(*parameters, preset_name)
    results.append((accuracy_history[-1], preset_name))
    
    plt.plot(accuracy_history, label=preset_name)
    
results.sort(reverse=True)
for result in results:
    print(f'Preset: {result[1]}, Final accuracy: {result[0]}')

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('cat&dog_accuracy.png')
#plt.show()
