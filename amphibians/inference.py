import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import json
import os

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

# Global threshold variable
THRESHOLD = 0.5

def get_model(num_classes):
    #model = fasterrcnn_resnet50_fpn_v2(weights=None)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_model(model_path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, image_path, device='cuda'):
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # Perform inference
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
            draw.rectangle(
                [box[0].item(), box[1].item(), box[2].item(), box[3].item()],
                outline="red",
                width=2
            )
            
            text = f"{label_mapping[label.item()]}: {score.item():.2f}"
            text_width, text_height = draw.textsize(text, font=font)
            text_position = (box[0].item(), box[1].item())
            
            # Draw a rectangle behind the text for better visibility (optional)
            draw.rectangle(
                [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
                fill="red"
            )
            
            # Write the text on the image
            draw.text(text_position, text, fill="white", font=font)
    
    # Save the image with drawn bounding boxes and text
    image.save(save_path)


def process_directory(test_dir, model_path, output_dir):
    num_classes = 17  # Including the background class
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load the trained model
    model = load_model(model_path, num_classes)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the directory
    for subdir, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(subdir, file)
                save_path = os.path.join(output_dir, os.path.relpath(image_path, start=test_dir))
                
                # Ensure subdirectories in the output directory exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                print(f"Processing {image_path}")
                
                # Perform predictions
                predictions = predict(model, image_path, device)

                # Map predictions to labels
                predicted_boxes = []
                for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
                    if score >= THRESHOLD:  # Use the global threshold variable
                        predicted_boxes.append({
                            "xmin": box[0].item(),
                            "ymin": box[1].item(),
                            "xmax": box[2].item(),
                            "ymax": box[3].item(),
                            "label": label_mapping[label.item()],
                            "score": score.item()
                        })

                # Print the predicted boxes
                #print(json.dumps(predicted_boxes, indent=4, ensure_ascii=False))
                # Visualize and save predictions
                visualize_predictions(image_path, predictions, save_path)

if __name__ == "__main__":
    test_dir = 'datasets/amphibia/test'  # Update with the path to your test folder
    model_path = 'amphibians/models/best_model.pth'  # Update with the path to your trained model
    output_dir = 'amphibians/evaluation'  # Update with the path to your output folder
    process_directory(test_dir, model_path, output_dir)
