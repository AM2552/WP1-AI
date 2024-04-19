from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# datasetname/
#     training/
#         cat/
#         dog/
#     validation/
#         cat/
#         dog/

train_generator = train_datagen.flow_from_directory(
        'datasets/cat_vs_dog/training',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        'datasets/cat_vs_dog/validation',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')
