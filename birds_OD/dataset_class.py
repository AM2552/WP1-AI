import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class BirdsDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transforms=None):
        with open(annotations_file) as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_paths = self._get_img_paths()

    def _get_img_paths(self):
        img_paths = []
        for root, _, files in os.walk(self.img_dir):
            for file in files:
                if file.endswith('.jpg'):
                    img_paths.append(os.path.join(root, file))
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_filename = os.path.relpath(img_path, self.img_dir).replace('\\', '/')
        
        # Find the corresponding annotation
        ann = next((a for a in self.annotations if a['image_path'] == img_filename), None)
        if ann is None:
            raise FileNotFoundError(f"No annotation found for {img_filename}")

        image = Image.open(img_path).convert("RGB")
        
        boxes = ann['bbox']
        # Ensure all bounding boxes have positive width and height
        x_min, y_min, x_max, y_max = boxes['x_min'], boxes['y_min'], boxes['x_max'], boxes['y_max']
        width = x_max - x_min
        height = y_max - y_min
        if width > 0 and height > 0:
            valid_boxes = [[x_min, y_min, x_max, y_max]]
        else:
            valid_boxes = []

        boxes = torch.as_tensor(valid_boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, target