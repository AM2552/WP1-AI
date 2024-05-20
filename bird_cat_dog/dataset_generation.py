from keras.preprocessing.image import ImageDataGenerator

# datasetname/
#     training/
#         bird/
#         cat/
#         dog/
#     validation/
#         bird/
#         cat/
#         dog/

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=35,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
        'datasets/bird_cat_dog/training',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
        'datasets/bird_cat_dog/validation',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')
