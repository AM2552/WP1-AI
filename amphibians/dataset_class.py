import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import random
import math

class AmphibianDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transforms=None, augmentations=None):
        self.annotations_file = annotations_file
        self.img_dir = img_dir
        self.transforms = transforms
        self.augmentations = augmentations
        self.annotations = self._load_annotations()
        self.img_paths = self._get_img_paths()
        print(f"Found {len(self.img_paths)} images in {img_dir}")

    def _load_annotations(self):
        tree = ET.parse(self.annotations_file)
        root = tree.getroot()

        annotations = {}
        for image in root.findall('image'):
            img_name = image.get('name')
            boxes = []
            for box in image.findall('box'):
                label = box.get('label')
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                boxes.append({
                    'label': label,
                    'bbox': [xtl, ytl, xbr, ybr]
                })
            annotations[img_name] = boxes
        return annotations

    def _get_img_paths(self):
        img_paths = []
        for root, _, files in os.walk(self.img_dir):
            for file in files:
                if file.endswith(('.jpg', '.png')):
                    img_paths.append(os.path.join(root, file))
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_filename = os.path.basename(img_path)
        
        # Find the corresponding annotation
        ann = self.annotations.get(img_filename, [])
        
        image = Image.open(img_path).convert("RGB")
        
        boxes = []
        labels = []
        for bbox_info in ann:
            bbox = bbox_info['bbox']
            boxes.append(bbox)
            # Convert label name to a label index
            label_name = bbox_info['label']
            label_idx = [k for k, v in label_mapping.items() if v == label_name]
            if label_idx:
                labels.append(label_idx[0])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        # Apply augmentations
        if self.augmentations:
            image, target = self.augmentations(image, target)
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, target

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

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image, target):
        if random.random() < self.probability:
            image = T.functional.hflip(image)
            w, h = image.size
            boxes = target['boxes']
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target['boxes'] = boxes
        return image, target

class RandomVerticalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image, target):
        if random.random() < self.probability:
            image = T.functional.vflip(image)
            w, h = image.size
            boxes = target['boxes']
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
            target['boxes'] = boxes
        return image, target

class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, target):
        angle = random.uniform(-self.degrees, self.degrees)
        image = T.functional.rotate(image, angle)
        w, h = image.size
        boxes = target['boxes']
        cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        new_boxes = []
        angle_rad = angle * math.pi / 180  # Convert angle to radians
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            points = torch.tensor([
                [x1, y1],
                [x1, y2],
                [x2, y1],
                [x2, y2]
            ], dtype=torch.float32)
            points = points - torch.tensor([[cx[i], cy[i]]])
            theta = torch.tensor([[cos_angle, -sin_angle],
                                  [sin_angle, cos_angle]])
            new_points = torch.matmul(points, theta) + torch.tensor([[cx[i], cy[i]]])
            new_boxes.append([
                new_points[:, 0].min().item(), new_points[:, 1].min().item(),
                new_points[:, 0].max().item(), new_points[:, 1].max().item()
            ])
        target['boxes'] = torch.tensor(new_boxes, dtype=torch.float32)
        return image, target