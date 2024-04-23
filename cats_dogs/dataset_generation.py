from keras.preprocessing.image import ImageDataGenerator

# datasetname/
#     training/
#         cat/
#         dog/
#     validation/
#         cat/
#         dog/

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
        'datasets/cat_vs_dog/training',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
        'datasets/cat_vs_dog/validation',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')
