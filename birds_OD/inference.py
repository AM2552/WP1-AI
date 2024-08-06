import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
import os

# Function to get the model and load trained weights
def get_model(num_classes, model_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to get all image paths from a directory with subfolders
def get_all_image_paths(test_dir):
    image_paths = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Function to process a single image and return the prediction
def predict(model, image, device):
    transform = T.Compose([T.ToTensor()])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image)
    return prediction

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, labels, scores, threshold=0.5):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red", width=2)
            draw.text((box[0], box[1]), f"{label}: {score:.2f}", fill="red")
    return image

# Inference function to process images from the test directory and save results
def run_inference(model, test_dir, output_dir, device, threshold=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = get_all_image_paths(test_dir)

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        predictions = predict(model, image, device)

        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        image_with_boxes = draw_boxes(image, boxes, labels, scores, threshold)
        output_path = os.path.join(output_dir, os.path.relpath(image_path, test_dir))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image_with_boxes.save(output_path)
        print(f"Processed and saved: {output_path}")

def main():
    test_dir = 'datasets/birds/validation'
    output_dir = 'birds_OD/evaluation'
    model_path = 'birds_OD/models/frcnn_e5.pth'
    num_classes = 201  # 200 bird classes + 1 background

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = get_model(num_classes, model_path)
    model.to(device)

    run_inference(model, test_dir, output_dir, device)

if __name__ == "__main__":
    main()
