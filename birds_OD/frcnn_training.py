import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import torchvision.transforms as T
from dataset_class import BirdsDataset
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Custom collate function to handle the batching
def collate_fn(batch):
    return tuple(zip(*batch))

# Training loop
def train(model, data_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0.0
    pbar = tqdm(data_loader, desc=f'Epoch {epoch+1}', leave=True)
    for images, targets in pbar:
        # Filter out images with empty bounding boxes
        filtered = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt['boxes']) > 0]
        if not filtered:
            continue
        
        images, targets = zip(*filtered)

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()
        pbar.set_postfix(loss=train_loss / len(data_loader))
    return train_loss / len(data_loader)

# Evaluation loop
def evaluate(model, data_loader, device, epoch):
    model.eval()
    eval_loss = 0.0
    pbar = tqdm(data_loader, desc=f'Eval {epoch+1}', leave=True)
    with torch.no_grad():
        for images, targets in pbar:
            # Filter out images with empty bounding boxes
            filtered = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt['boxes']) > 0]
            if not filtered:
                continue
            
            images, targets = zip(*filtered)

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model.train()  # Switch to training mode to compute loss
            eval_loss_dict = model(images, targets)
            model.eval()  # Switch back to evaluation mode
            eval_losses = sum(loss for loss in eval_loss_dict.values())
            
            eval_loss += eval_losses.item()
            pbar.set_postfix(loss=eval_loss / len(data_loader))
    return eval_loss / len(data_loader)
# GPU selector
def get_free_gpu():
    free_memory = []
    for i in range(torch.cuda.device_count()):
        info = torch.cuda.memory_reserved(i)
        free_memory.append((i, torch.cuda.get_device_properties(i).total_memory - info))
    return sorted(free_memory, key=lambda x: x[1], reverse=True)[0][0]

def main():
    train_img_dir = 'datasets/birds/training'
    val_img_dir = 'datasets/birds/validation'
    annotations_file = 'birds_OD/birds_annotations.json'
    
    num_classes = 201  # 200 bird classes + 1 background

    # Define the transformations
    transform = T.Compose([
        T.ToTensor(),
    ])

    train_dataset = BirdsDataset(annotations_file, train_img_dir, transforms=transform)
    val_dataset = BirdsDataset(annotations_file, val_img_dir, transforms=transform)
    
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Select the GPU with the most available memory
    if torch.cuda.is_available():
        device_id = get_free_gpu()
        device = torch.device(f'cuda:{device_id}')
        print(f'Using GPU: {device_id}')
    else:
        device = torch.device('cpu')

    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)

    num_epochs = 5
    train_losses = []
    eval_losses = []
    
    for epoch in range(num_epochs):
        gc.collect()  # Collect garbage to free memory
        torch.cuda.empty_cache()  # Free up unused cached memory
        
        train_loss = train(model, train_data_loader, optimizer, device, epoch)
        eval_loss = evaluate(model, val_data_loader, device, epoch)
        
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
    
        torch.save(model.state_dict(), f"birds_OD/models/frcnn_e{epoch+1}.pth")
    
    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), eval_losses, label='Eval Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("frcnn_adam_1e-4.png")
    plt.show()

if __name__ == "__main__":
    main()
