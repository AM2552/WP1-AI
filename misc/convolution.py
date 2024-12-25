import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def apply_vertical_edge_detection_color(image_path):
    # Load the image
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    input_image = transform(image).unsqueeze(0)  # Shape: (1, C, H, W)

    # Apply convolution to each channel independently
    vertical_edge_kernel = torch.tensor([[1, 1, 1],
                                         [0, 0, 0],
                                         [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)

    # Apply the kernel to each channel
    channels = []
    for c in range(input_image.size(1)):  # Loop over color channels
        channel = input_image[:, c:c+1, :, :]  # Isolate channel
        edge_detected = F.conv2d(channel, vertical_edge_kernel, padding=1)
        channels.append(edge_detected)

    # Combine channels back
    output = torch.cat(channels, dim=1)  # Combine into (1, C, H, W)

    # Normalize for visualization
    output = (output - output.min()) / (output.max() - output.min())
    output_image = transforms.ToPILImage()(output.squeeze(0))  # Convert to image

    # Display input and output images
    plt.figure(figsize=(5, 5))
    plt.title("Horizontal Edges")
    plt.imshow(output_image)
    plt.axis('off')

    #plt.subplot(1, 2, 2)
    #plt.title("")
    #plt.imshow(output_image)
    #plt.axis('off')

    plt.show()

# Provide the path to your input image
apply_vertical_edge_detection_color("misc/image.jpg")
