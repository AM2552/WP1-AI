import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.mobile_optimizer import optimize_for_mobile

# Load your trained model
model = fasterrcnn_resnet50_fpn_v2(weights=None)

# Replace the box predictor with a new one (note the use of FastRCNNPredictor)
num_classes = 17  # Set this to the number of classes in your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the state dictionary
model.load_state_dict(torch.load("amphibians/models/best_model_resnet_v2_pretrained.pth"))
model.eval()

# Convert to TorchScript
scripted_model = torch.jit.script(model)
optimized_model = optimize_for_mobile(scripted_model)
optimized_model._save_for_lite_interpreter("model.ptl")
