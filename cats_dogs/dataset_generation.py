from keras.preprocessing.image import ImageDataGenerator

# Create ImageDataGenerators for training and validation datasets
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Assuming your dataset is in a directory with a structure:
# dataset/
#     train/
#         cats/
#         dogs/
#     validation/
#         cats/
#         dogs/

train_generator = train_datagen.flow_from_directory(
        'datasets/cats_vs_dogs/training',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        'datasets/cats_vs_dogs/validation',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')
