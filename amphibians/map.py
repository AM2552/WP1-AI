import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import os
import json
from torchmetrics.detection.mean_ap import MeanAveragePrecision

label_mapping = {
    1: 'feuersalamander',
    2: 'alpensalamander',
    3: 'bergmolch',
    4: 'kammmolch',
    5: 'teichmolch',
    6: 'rotbauchunke',
    7: 'gelbbauchunke',
    8: 'knoblauchkröte',
    9: 'erdkröte',
    10: 'kreuzkröte',
    11: 'wechselkröte',
    12: 'laubfrosch',
    13: 'moorfrosch',
    14: 'springfrosch',
    15: 'grasfrosch',
    16: 'wasserfrosch'
}

THRESHOLD = 0.5

def get_model(num_classes):
    #model = fasterrcnn_resnet50_fpn_v2(weights=None)
    #model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_model(model_path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    return model

def predict(model, image_path, device='cuda'):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    model.to(device)
    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions

def visualize_predictions(image_path, predictions, save_path, threshold=THRESHOLD):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    font_size = 15
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if score >= threshold:
            if label.item() == 0:
                continue  # Skip background class

            # Adjust label index for mapping (since ground truth labels are from 1 to 16)
            label_idx = label.item()
            if label_idx not in label_mapping:
                label_name = f"Unknown_{label_idx}"
            else:
                label_name = label_mapping[label_idx]

            draw.rectangle(
                [box[0].item(), box[1].item(), box[2].item(), box[3].item()],
                outline="red",
                width=2
            )

            text = f"{label_name}: {score.item():.2f}"
            text_width, text_height = draw.textsize(text, font=font)
            text_position = (box[0].item(), box[1].item())

            draw.rectangle(
                [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
                fill="red"
            )
            draw.text(text_position, text, fill="white", font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)

def load_annotations(annotations_file):
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    gt_boxes = {}
    for item in annotations:
        image_filename = os.path.basename(item['image_path'])
        bboxes = item['bboxes']
        gt_boxes[image_filename] = bboxes
    return gt_boxes

def process_directory(test_dir, model_path, annotations_file, output_dir):
    num_classes = 17  # Including background as class 0
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(model_path, num_classes)

    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)

    gt_annotations = load_annotations(annotations_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(subdir, file)
                image_filename = os.path.basename(image_path)
                save_path = os.path.join(output_dir, os.path.relpath(image_path, start=test_dir))

                print(f"Processing {image_path}")

                # Get ground truth
                if image_filename in gt_annotations:
                    gt_bboxes = gt_annotations[image_filename]
                    gt_boxes_list = []
                    gt_labels_list = []
                    for bbox in gt_bboxes:
                        label = bbox['class_label']
                        bbox_coords = bbox['bbox']
                        gt_boxes_list.append([
                            bbox_coords['x_min'],
                            bbox_coords['y_min'],
                            bbox_coords['x_max'],
                            bbox_coords['y_max']
                        ])
                        gt_labels_list.append(label)
                    gt_boxes_tensor = torch.tensor(gt_boxes_list, dtype=torch.float32)
                    # Adjust labels to start from 0
                    gt_labels_tensor = torch.tensor(gt_labels_list, dtype=torch.int64)
                    target = {'boxes': gt_boxes_tensor, 'labels': gt_labels_tensor}
                else:
                    # No ground truth for this image
                    target = {'boxes': torch.zeros((0, 4)), 'labels': torch.zeros((0,), dtype=torch.int64)}

                # Run prediction
                predictions = predict(model, image_path, device)

                # Process predictions
                pred_boxes_list = []
                pred_labels_list = []
                pred_scores_list = []
                for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
                    if label.item() == 0:
                        continue  # Skip background class
                    pred_boxes_list.append(box.cpu())
                    # Adjust labels to start from 0
                    pred_labels_list.append(label.cpu())
                    pred_scores_list.append(score.cpu())

                if len(pred_boxes_list) > 0:
                    pred_boxes_tensor = torch.stack(pred_boxes_list)
                    pred_labels_tensor = torch.stack(pred_labels_list)
                    pred_scores_tensor = torch.stack(pred_scores_list)
                else:
                    pred_boxes_tensor = torch.zeros((0, 4))
                    pred_labels_tensor = torch.zeros((0,), dtype=torch.int64)
                    pred_scores_tensor = torch.zeros((0,))

                preds = {
                    'boxes': pred_boxes_tensor,
                    'labels': pred_labels_tensor,
                    'scores': pred_scores_tensor
                }

                # Update metric
                metric.update([preds], [target])

                # Visualize and save predictions
                visualize_predictions(image_path, predictions, save_path)

    # Compute mAP
    mAP_result = metric.compute()

    # Process and print per-class mAP
    print("\nMean Average Precision results:")
    print(f"Overall mAP: {mAP_result['map'].item():.4f}")
    print(f"mAP@0.5: {mAP_result['map_50'].item():.4f}")
    print(f"mAP@0.75: {mAP_result['map_75'].item():.4f}\n")

    if 'map_per_class' in mAP_result and 'mar_100_per_class' in mAP_result:
        map_per_class = mAP_result['map_per_class']
        mar_100_per_class = mAP_result['mar_100_per_class']

        # Since we adjusted labels to start from 0, we can map them directly
        classes = [label_mapping[i + 1] for i in range(len(label_mapping))]

        print("Per-class Mean Average Precision (mAP):")
        for idx, class_name in enumerate(classes):
            ap = map_per_class[idx].item()
            ar = mar_100_per_class[idx].item()
            print(f"Class '{class_name}': AP={ap:.4f}, AR={ar:.4f}")

if __name__ == "__main__":
    test_dir = 'datasets/amphibia/test'
    model_path = 'amphibians/models/model_epoch_145.pth'
    annotations_file = 'amphibians/amphibia_annotations.json'  # Path to your annotations file
    output_dir = 'amphibians/evaluation'  # Directory to save visualized images
    process_directory(test_dir, model_path, annotations_file, output_dir)
