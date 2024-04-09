from keras.preprocessing.image import ImageDataGenerator

# Create ImageDataGenerators for training and validation datasets
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Assuming your dataset is in a directory with a structure:
# datasetname/
#     training/
#         cat/
#         dog/
#     validation/
#         cat/
#         dog/

train_generator = train_datagen.flow_from_directory(
        'datasets/cat&dog/training',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        'datasets/cat&dog/validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')
