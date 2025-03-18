import numpy as np
from PIL import Image

import tflite_runtime.interpreter as tflite

# If using the TF-lite runtime standalone, you do:
# interpreter = tflite.Interpreter(model_path="converted.tflite")
# If using full TensorFlow:
# interpreter = tf.lite.Interpreter(model_path="converted.tflite")

# 1) Load the TFLite model
model_path = "amphibians/yolo_training/best_saved_model/best_float32.tflite"

# Attempt full TensorFlow first, fallback to tflite_runtime

interpreter = tflite.Interpreter(model_path=model_path)

# 2) Allocate tensors (must be done before setting/getting tensors)
interpreter.allocate_tensors()

# 3) Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

img = Image.open("datasets/amphibia/test/alpensalamander/0deea065-47ab-4409-8d81-91614576d18e.jpg").convert("RGB")
img = img.resize((640, 640))
input_array = np.array(img, dtype=np.float32) / 255.0  # if your model is normalized
# shape: (640, 640, 3)

# expand dims to [1, 640, 640, 3]
input_array = np.expand_dims(input_array, axis=0)

interpreter.set_tensor(input_details[0]['index'], input_array)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])  # (1, 20, 8400)
predictions = output_data[0].transpose(1, 0)  # => shape (8400, 20)

for i in range(8400):
    box = predictions[i]  # shape (20,)
    x_center, y_center, w, h = box[0:4]
    obj_conf = box[4]
    class_probs = box[5:]  # shape (15,) if 15 classes
    if obj_conf > 0.2:
        print(f"Box {i}: {x_center}, {y_center}, {w}, {h}, {obj_conf}, {class_probs}")
