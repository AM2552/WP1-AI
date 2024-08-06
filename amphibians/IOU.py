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

# Load the saved weights
def load_trained_model(model_path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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

def draw_boxes(image, gt_boxes, pred_boxes, pred_labels, scores):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box in gt_boxes:
        draw.rectangle(box, outline="green", width=2)

    for box, label, score in zip(pred_boxes, pred_labels, scores):
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{label}:{score:.2f}", fill="red", font=font)

    return image

def evaluate_model_on_test(model, test_dir, annotations, device, threshold=0.1):
    total_iou = 0.0
    total_gt_boxes = 0
    total_pred_boxes = 0
    results = []

    # Get all image paths considering subfolders
    img_paths = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                img_paths.append(os.path.join(root, file))

    for img_path in tqdm(img_paths):
        img_filename = os.path.basename(img_path)
        
        # Find the corresponding annotation
        ann = next((a for a in annotations if a['image_path'] == img_filename), None)
        if ann is None:
            raise FileNotFoundError(f"No annotation found for {img_filename}")

        image = Image.open(img_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)

        predicted_boxes = outputs[0]['boxes'].cpu().numpy()
        predicted_labels = outputs[0]['labels'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        valid_indices = [i for i, score in enumerate(scores) if score > threshold]
        predicted_boxes = predicted_boxes[valid_indices]
        predicted_labels = predicted_labels[valid_indices]
        scores = scores[valid_indices]

        gt_boxes = []
        for bbox_info in ann['bboxes']:
            gt_boxes.append([
                bbox_info['bbox']['x_min'],
                bbox_info['bbox']['y_min'],
                bbox_info['bbox']['x_max'],
                bbox_info['bbox']['y_max']
            ])

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
            'filename': img_filename,
            'gt_boxes': gt_boxes,
            'pred_boxes': predicted_boxes.tolist(),
            'iou': image_iou / len(gt_boxes) if gt_boxes else 0
        })

        # Draw and save the image with boxes
        annotated_image = draw_boxes(image.copy(), gt_boxes, predicted_boxes, predicted_labels, scores)
        save_path = f"amphibians/evaluation/{img_filename}_predicted.jpg"
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
    model_path = "amphibians/models/frcnn_e1.pth"
    annotations_path = "amphibians/amphibia_annotations.json"
    test_dir = "datasets/amphibia/test"
    num_classes = 17  # Update this based on your dataset

    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_trained_model(model_path, num_classes)

    if torch.cuda.is_available():
        device_id = get_free_gpu()
        device = torch.device(f'cuda:{device_id}')
        print(f'Using GPU: {device_id}')
    else:
        device = torch.device('cpu')
    model.to(device)

    average_iou, results = evaluate_model_on_test(model, test_dir, annotations, device)
    print(f"Average IoU: {average_iou:.4f}")

    for result in results:
        print(f"Filename: {result['filename']}, IoU: {result['iou']:.4f}")

if __name__ == "__main__":
    main()
