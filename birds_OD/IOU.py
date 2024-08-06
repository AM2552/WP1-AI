import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Build the model
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# load the saved weights
def load_trained_model(model_path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def draw_boxes(image, gt_boxes, pred_boxes, scores):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box in gt_boxes:
        draw.rectangle(box, outline="green", width=2)

    for box, score in zip(pred_boxes, scores):
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{score:.2f}", fill="red", font=font)

    return image

def evaluate_model_on_test(model, test_data, device, threshold=0.7):
    total_iou = 0.0
    total_gt_boxes = 0
    total_pred_boxes = 0
    results = []

    for data in tqdm(test_data):
        image_path = os.path.join("test_images", data['filename'])
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)

        predicted_boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        valid_indices = [i for i, score in enumerate(scores) if score > threshold]
        predicted_boxes = predicted_boxes[valid_indices]
        scores = scores[valid_indices]

        gt_boxes = []
        for bbox in data['bboxes']:
            x_min = bbox['x']
            y_min = bbox['y']
            x_max = x_min + bbox['width']
            y_max = y_min + bbox['height']
            gt_boxes.append([x_min, y_min, x_max, y_max])

        image_iou = 0.0
        for gt_box in gt_boxes:
            best_iou = 0.0
            for pred_box in predicted_boxes:
                iou = calculate_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
            image_iou += best_iou

        total_iou += image_iou
        total_gt_boxes += len(gt_boxes)
        total_pred_boxes += len(predicted_boxes)
        
        results.append({
            'filename': data['filename'],
            'gt_boxes': gt_boxes,
            'pred_boxes': predicted_boxes.tolist(),
            'iou': image_iou / len(gt_boxes) if gt_boxes else 0
        })

        # Draw and save the image with boxes
        annotated_image = draw_boxes(image.copy(), gt_boxes, predicted_boxes, scores)
        save_path = os.path.join("test_images", f"{data['filename']}_predicted.jpg")
        annotated_image.save(save_path)

    average_iou = total_iou / total_gt_boxes if total_gt_boxes > 0 else 0
    return average_iou, results

# GPU selector
def get_free_gpu():
    free_memory = []
    for i in range(torch.cuda.device_count()):
        info = torch.cuda.memory_reserved(i)
        free_memory.append((i, torch.cuda.get_device_properties(i).total_memory - info))
    return sorted(free_memory, key=lambda x: x[1], reverse=True)[0][0]

def main():
    model_path = "./models/resnet50_e5_adamw.pth"
    test_json_path = "./test_images/wider_face_test.json"
    num_classes = 2 

    with open(test_json_path) as f:
        test_data = json.load(f)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_trained_model(model_path, num_classes)

    if torch.cuda.is_available():
        device_id = get_free_gpu()
        device = torch.device(f'cuda:{device_id}')
        print(f'Using GPU: {device_id}')
    else:
        device = torch.device('cpu')
    model.to(device)

    average_iou, results = evaluate_model_on_test(model, test_data, device)
    print(f"Average IoU: {average_iou:.4f}")

    for result in results:
        print(f"Filename: {result['filename']}, IoU: {result['iou']:.4f}")

if __name__ == "__main__":
    main()
