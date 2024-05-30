import imageio
import os

# Directory containing the images
image_dir = 'bird_cat_dog/test_images/grad_cam/cat'

# Parameters for the video
output_video = 'grad_cam_video.mp4'
frame_rate = 10  # frames per second

# Get a list of image files
images = [img for img in os.listdir(image_dir) if img.endswith(".jpg")]
images.sort()  # Ensure the images are sorted correctly

# Create a video writer object
writer = imageio.get_writer(output_video, fps=frame_rate)

# Add each image to the video
for image in images:
    image_path = os.path.join(image_dir, image)
    img = imageio.imread(image_path)
    writer.append_data(img)

# Close the writer object
writer.close()

print(f"Video saved as {output_video}")