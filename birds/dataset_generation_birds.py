import tensorflow as tf
from tensorflow import keras
from keras.layers import Resizing, Rescaling, RandomFlip, RandomRotation, RandomZoom, RandomWidth, RandomHeight, CenterCrop
import os

# Define the function to resize and center crop images
def preprocess_image(target_size=(256, 256), crop_size=(240, 240)):
    preprocessing_layers = tf.keras.Sequential([
        Resizing(target_size[0], target_size[1]),
        CenterCrop(crop_size[0], crop_size[1]),
        Rescaling(1./255)
    ])
    return preprocessing_layers

# Load the dataset and apply preprocessing and augmentation
def load_datasets(train_dir, validation_dir, batch_size=32, img_size=(240, 240)):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )
    
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    # Define the augmentation layers
    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.1),
        RandomWidth(0.1),
        RandomHeight(0.1)
    ])

    # Apply preprocessing and augmentation to the training dataset
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

    # Apply only preprocessing to the validation dataset
    preprocessing_layers = preprocess_image()
    train_dataset = train_dataset.map(lambda x, y: (preprocessing_layers(x), y))
    validation_dataset = validation_dataset.map(lambda x, y: (preprocessing_layers(x), y))
    
    return train_dataset, validation_dataset

# Define the directories for the training and validation datasets
train_dir = 'datasets/birds/training'
validation_dir = 'datasets/birds/validation'

# Load datasets
train_dataset, validation_dataset = load_datasets(train_dir, validation_dir)
