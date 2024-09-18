import tensorflow as tf


h5_model_path = 'bird_cat_dog_model_adam_eff1.h5'
model = tf.keras.models.load_model(h5_model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_model_path = 'bcd_eff1.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model successfully converted!")