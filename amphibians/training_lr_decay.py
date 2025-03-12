import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    fasterrcnn_resnet50_fpn_v2, 
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    ResNet50_Weights,
    MobileNet_V3_Large_Weights
)
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from dataset_class import AmphibianDataset, Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
import torchvision.transforms as T
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1,
        trainable_backbone_layers=6
        )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Training loop
def train(model, data_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0.0
    pbar = tqdm(data_loader, desc=f'Epoch {epoch+1}', leave=True)
    for images, targets in pbar:
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
            filtered = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt['boxes']) > 0]
            if not filtered:
                continue
            
            images, targets = zip(*filtered)

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Temporarily set model to train to compute the loss in evaluation
            model.train()
            eval_loss_dict = model(images, targets)
            model.eval()

            eval_losses = sum(loss for loss in eval_loss_dict.values())
            eval_loss += eval_losses.item()
            pbar.set_postfix(loss=eval_loss / len(data_loader))
    return eval_loss / len(data_loader)

def plot_losses(train_losses, val_losses, save_dir):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.show()

def main(pretrained_model_path):
    
    transforms = T.Compose([
        T.ToTensor()
    ])

    augmentations = Compose([
        RandomHorizontalFlip(probability=0.5),
        RandomVerticalFlip(probability=0.5),
        RandomRotation(degrees=30)
    ])

    train_dataset = AmphibianDataset(
        annotations_file='amphibians/amphibia_annotations.json',
        img_dir='datasets/amphibia/training',
        transforms=transforms,
        augmentations=augmentations
    )

    val_dataset = AmphibianDataset(
        annotations_file='amphibians/amphibia_annotations.json',
        img_dir='datasets/amphibia/validation',
        transforms=transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = get_model(num_classes=17)
    #state_dict = torch.load('amphibians/cnn/models/resnet50_adam_default.pth')
    #state_dict.pop("fc.1.weight", None)
    #state_dict.pop("fc.1.bias", None)
    #model.backbone.body.load_state_dict(state_dict)
    model.to('cuda')

    # HYPERPARAMETERS #
    num_epochs = 150
    initial_lr = 1e-4
    
    optimizer = Adam(model.parameters(), lr=initial_lr)
    exponentialLR_scheduler = ExponentialLR(optimizer, gamma=0.92)
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.75,
        patience=2,
        threshold=1e-3,
        min_lr=1e-6
    )
    
    scheduler = plateau_scheduler
    ###############


    # Load a pretrained checkpoint
    if pretrained_model_path is not None and os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path))

    save_dir = 'amphibians/models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_validation_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}/{num_epochs} (Current LR: {scheduler.get_last_lr()[0]:.6f})")

        train_loss = train(model, train_loader, optimizer, 'cuda', epoch)
        train_losses.append(train_loss)

        validation_loss = evaluate(model, val_loader, 'cuda', epoch)
        val_losses.append(validation_loss)

        print(f"Training Loss: {round(train_loss, 5)} / Validation Loss: {round(validation_loss, 5)}")

        scheduler.step(validation_loss)  # Update the learning rate

        # Save the best model
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model.pth"))
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))

    plot_losses(train_losses, val_losses, save_dir)

if __name__ == '__main__':
    pretrained_model_path = None
    main(pretrained_model_path=pretrained_model_path)
