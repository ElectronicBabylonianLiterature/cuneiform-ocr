import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

def show_image(image):
    inverted_image_tensor = torch.flip(image, dims=[2])
    # Convert the tensor to a numpy array and transpose it
    image_np = inverted_image_tensor.squeeze(0).permute(1, 2, 0).numpy()

    # Display the image
    plt.imshow(image_np)
    plt.show()