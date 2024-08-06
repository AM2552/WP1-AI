import tensorflow as tf
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

def resize_and_center_crop(image, target_size, crop_size):
       image = tf.image.resize(image, target_size)
       return tf.image.central_crop(image, crop_size[0] / image.shape[0])

def preprocess_image(image):
     target_size = (256, 256)
     crop_size = (240, 240)
     image = resize_and_center_crop(image, target_size, crop_size)
     return image

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=35,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_image
)

train_generator = train_datagen.flow_from_directory(
        'datasets/bird_cat_dog/training',
        target_size=(240, 240),
        batch_size=32,
        class_mode='categorical',
        shuffle=True)

validation_datagen = ImageDataGenerator(
      rescale=1./255,
      preprocessing_function=preprocess_image
      )

validation_generator = validation_datagen.flow_from_directory(
        'datasets/bird_cat_dog/validation',
        target_size=(240, 240),
        batch_size=32,
        class_mode='categorical')
