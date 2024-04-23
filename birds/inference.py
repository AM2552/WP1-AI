import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

# Load the model
model = torch.load('cats_vs_dogs_model_Preset3.pt')
model.eval()  # Set the model to evaluation mode

folder_path = './datasets/cat_vs_dog/test'
accuracy_counter = 0
dog_accuracy = 0
cat_accuracy = 0

# Prepare the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def pad_image(image_path):
    try:
        img = Image.open(image_path)
        width, height = img.size
        size = max(width, height)
        new_img = Image.new("RGB", (size, size))
        new_img.paste(img, ((size - width) // 2, (size - height) // 2))
        return new_img
    except OSError:
        print(f"Skipping file due to OSError: {image_path}")

dog_counter = len([name for name in os.listdir(os.path.join(folder_path, 'dog')) if os.path.isfile(os.path.join(folder_path, 'dog', name))])
cat_counter = len([name for name in os.listdir(os.path.join(folder_path, 'cat')) if os.path.isfile(os.path.join(folder_path, 'cat', name))])

for folder in tqdm(os.listdir(folder_path), desc="Processing classes"):
    for filename in tqdm(os.listdir(os.path.join(folder_path, folder)), desc="Processing images"):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, folder, filename)
            img = pad_image(img_path)
            img = img.resize((256, 256))
            img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                outputs = model(img_tensor)
                prediction = torch.sigmoid(outputs).item()
                class_name = "dog" if prediction > 0.5 else "cat"
                probability = prediction * 100 if class_name == "dog" else (1 - prediction) * 100
            
            if class_name == 'dog' and 'dog' in folder:
                dog_accuracy += 1
            elif class_name == 'cat' and 'cat' in folder:
                cat_accuracy += 1

total_images = dog_counter + cat_counter
dog_accuracy = dog_accuracy / dog_counter * 100
cat_accuracy = cat_accuracy / cat_counter * 100
val_accuracy = (dog_accuracy + cat_accuracy) / 2
with open('accuracy.txt', 'w') as f:
    f.write(f"Dog accuracy: {dog_accuracy:.2f}%\n")
    f.write(f"Cat accuracy: {cat_accuracy:.2f}%\n")
    f.write(f"Validation accuracy: {val_accuracy:.2f}%\n")

print(f"Validation accuracy: {val_accuracy:.2f}%")
