import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_mobilenet_v3_large_320_fpn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os
import json
import matplotlib.pyplot as plt
from PIL import Image

def get_model(num_classes):
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_model(model_path, num_classes, device):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_annotations(annotations_file):
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    gt_boxes = {}
    for item in annotations:
        image_filename = os.path.basename(item['image_path'])
        bboxes = item['bboxes']
        gt_boxes[image_filename] = bboxes
    return gt_boxes

def process_directory(test_dir, model, annotations_file, device):
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    gt_annotations = load_annotations(annotations_file)
    
    for subdir, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(subdir, file)
                image_filename = os.path.basename(image_path)
                
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
                    gt_labels_tensor = torch.tensor(gt_labels_list, dtype=torch.int64)
                    target = {'boxes': gt_boxes_tensor, 'labels': gt_labels_tensor}
                else:
                    target = {'boxes': torch.zeros((0, 4)), 'labels': torch.zeros((0,), dtype=torch.int64)}
                
                image = torch.unsqueeze(torchvision.transforms.functional.to_tensor(Image.open(image_path).convert("RGB")), 0).to(device)
                with torch.no_grad():
                    predictions = model(image)
                
                pred_boxes_list = []
                pred_labels_list = []
                pred_scores_list = []
                for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
                    if label.item() > 0:
                        pred_boxes_list.append(box.cpu())
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
                
                preds = {'boxes': pred_boxes_tensor, 'labels': pred_labels_tensor, 'scores': pred_scores_tensor}
                metric.update([preds], [target])
    
    return metric.compute()

def main():
    test_dir = 'datasets/amphibia/test'
    models_path = 'amphibians/models/'
    annotations_file = 'amphibians/amphibia_annotations.json'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    epochs = 25
    mAP_scores = []
    mAP_50_scores = []
    mAP_75_scores = []
    
    for i in range(1, epochs + 1):
        model_path = os.path.join(models_path, f"model_epoch_{i}.pth")
        if os.path.exists(model_path):
            print(f"Evaluating model: {model_path}")
            model = load_model(model_path, num_classes=17, device=device)
            mAP_result = process_directory(test_dir, model, annotations_file, device)
            
            mAP_scores.append(mAP_result['map'].item())
            mAP_50_scores.append(mAP_result['map_50'].item())
            mAP_75_scores.append(mAP_result['map_75'].item())
        else:
            print(f"Skipping missing model: {model_path}")
            mAP_scores.append(None)
            mAP_50_scores.append(None)
            mAP_75_scores.append(None)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), mAP_scores, label='Overall mAP')
    plt.plot(range(1, epochs + 1), mAP_50_scores, label='mAP@0.5')
    plt.plot(range(1, epochs + 1), mAP_75_scores, label='mAP@0.75')
    plt.xlabel('Epoch')
    plt.ylabel('mAP Score')
    plt.legend()
    plt.title('mAP Scores Over Epochs')
    plt.savefig('amphibians/models/mAP_scores.png')
    plt.show()
    

if __name__ == "__main__":
    main()
