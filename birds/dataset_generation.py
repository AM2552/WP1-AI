from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

# Define transformations for training
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),           # Resize images to 256x256
    transforms.RandomHorizontalFlip(),       # Randomly flip images horizontally
    transforms.RandomRotation(40),           # Random rotation of images
    transforms.RandomResizedCrop(256),       # Random crop then resize
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2, scale=(0.8, 1.2)),
                                             # Random affine transformation
    transforms.ToTensor(),                   # Convert images to tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize images
])

# Define transformations for validation
validation_transforms = transforms.Compose([
    transforms.Resize((256, 256)),           # Resize images to 256x256
    transforms.ToTensor(),                   # Convert images to tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize images
])

# Load datasets using ImageFolder
train_dataset = datasets.ImageFolder('datasets/birds/training', transform=train_transforms)
validation_dataset = datasets.ImageFolder('datasets/birds/validation', transform=validation_transforms)

# DataLoader to handle batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=4)