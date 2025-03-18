from ultralytics import YOLO

model = YOLO('amphibians/yolo_training/best.pt')

model.export(format='tflite')