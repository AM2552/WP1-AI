import os
from tqdm import tqdm
from PIL import Image

source_folder = '/Users/Xandi/Desktop/herpe/amphibien'

for folder in tqdm(os.listdir(source_folder), desc="Processing classes"):
    folder_path = os.path.join(source_folder, folder)
    if os.path.isdir(folder_path):
        counter = 1
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            new_filename = f'{folder}_{counter}.png'
            new_file_path = os.path.join(folder_path, new_filename)
            
            # Check if the new file already exists
            while os.path.exists(new_file_path):
                counter += 1
                new_filename = f'{folder}_{counter}.png'
                new_file_path = os.path.join(folder_path, new_filename)
            
            if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                if filename.endswith(('.jpg', '.jpeg', '.webp')):
                    with Image.open(file_path) as image:
                        # Convert CMYK images to RGB
                        if image.mode == 'CMYK':
                            image = image.convert('RGB')
                        image.save(new_file_path, 'PNG')
                    os.remove(file_path)  # Ensure the file handle is closed before removing
                else:  # If the file is already a PNG
                    os.rename(file_path, new_file_path)
                
                counter += 1
