import numpy as np
from PIL import ImageOps
from mmcv import BaseTransform
from mmengine import TRANSFORMS
from torchvision.transforms import Grayscale as Grayscale_
from PIL import Image

@TRANSFORMS.register_module()
class Grayscale(BaseTransform):
    def __init__(self):
        super().__init__()

    def transform(self, results):
        img =Image.fromarray(results["img"])
        img_grayscale = ImageOps.grayscale(img)
        image = np.array(img_grayscale)
        image = np.repeat(np.expand_dims(image, 2), 3, axis=2)
        results["img"] = image
        return results

