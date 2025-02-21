import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bars

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentation and normalization for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(
        degrees=35, 
        translate=(0.1, 0.1), 
        shear=10, 
        scale=(0.9, 1.1),
        fill=0
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# For validation, only resize and normalize
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Load datasets from directory structure
train_dataset = datasets.ImageFolder('datasets/amphibia/training', transform=train_transform)
val_dataset = datasets.ImageFolder('datasets/amphibia/validation', transform=val_transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

def train_model(learning_rate, optimizer_choice, epochs, preset_name):
    """
    Trains a ResNet50 classifier on the amphibian dataset.
    
    Args:
        learning_rate (float): Learning rate for the optimizer.
        optimizer_choice (str): 'adam' or 'sgd'.
        epochs (int): Number of training epochs.
        preset_name (str): Name identifier used for saving models and plots.
    
    Returns:
        Tuple of lists: (train_accuracy_history, val_accuracy_history, train_loss_history, val_loss_history)
    """
    # Load pretrained ResNet50 and modify the final layer
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 16)
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    if optimizer_choice == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    else:
        raise ValueError("Unsupported optimizer. Choose 'adam' or 'sgd'.")
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    train_acc_history, train_loss_history = [], []
    val_acc_history, val_loss_history = [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0

        # Training loop with progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
            
            # Update progress bar description
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())
        
        # Validation loop with progress bar
        model.eval()
        val_loss, val_corrects, total_val = 0.0, 0, 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation", leave=False)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                total_val += inputs.size(0)
                val_bar.set_postfix(loss=f"{loss.item():.4f}")
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_corrects.double() / total_val
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc.item())
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        scheduler.step()
    
    # Save the model
    model_path = f"amphibians/cnn/models/resnet50_{preset_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return train_acc_history, val_acc_history, train_loss_history, val_loss_history

def main():
    results = []
    # Dictionary mapping preset names to (learning_rate, optimizer, epochs)
    parameter_presets = {
        'adam_default': (0.001, 'adam', 50),
        # You can add more presets if needed
        # 'sgd_default': (0.005, 'sgd', 50),
    }
    
    for preset_name, parameters in parameter_presets.items():
        torch.cuda.empty_cache()
        train_acc_history, val_acc_history, train_loss_history, val_loss_history = train_model(
            *parameters, preset_name=preset_name)
        results.append((val_acc_history[-1], preset_name))
        
        # Plot accuracy history
        plt.figure(figsize=(10, 5))
        plt.plot(train_acc_history, label='Train Accuracy')
        plt.plot(val_acc_history, label='Validation Accuracy')
        plt.title(f"{preset_name} Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"amphibians/cnn/models/resnet50_{preset_name}_accuracy.png")
        plt.close()
        
        # Plot loss history
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history, label='Train Loss')
        plt.plot(val_loss_history, label='Validation Loss')
        plt.title(f"{preset_name} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"amphibians/cnn/models/resnet50_{preset_name}_loss.png")
        plt.close()
    
    results.sort(reverse=True)
    for final_val_acc, preset in results:
        print(f"Preset: {preset}, Final validation accuracy: {final_val_acc:.4f}")

if __name__ == '__main__':
    main()
