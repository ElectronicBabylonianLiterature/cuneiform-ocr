import torchvision
from PIL import Image
from matplotlib import pyplot as plt

# load image from path
path = "/home/yunus/PycharmProjects/cuneiform-ocr-3/data/coco/val2017/P335946-0.jpg"
image = Image.open(path)
# turn into python tensor
image = torchvision.transforms.functional.to_tensor(image).float()
plt.imshow(image.permute(1, 2, 0))
plt.show()
