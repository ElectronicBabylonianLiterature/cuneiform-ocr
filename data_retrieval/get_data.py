from numpy import absolute
from pymongo import MongoClient
import fragments
import json
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import crop
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.io import read_image
import torch
from torchvision.utils import draw_bounding_boxes
from PIL import ImageDraw
def line_boundaries(xs, ys, ws, hs):
    x = min(xs)
    y = min(ys)
    w = sum(ws)
    h = max(hs)
    return x, y, w, h




if __name__ == '__main__':
    client  = MongoClient(fragments.CONNECTION)
    database = client["ebl"]
    fragmentarium = database["fragments"]
    annotations = database["annotations"]
    xs, ys, ws, hs = [], [], [], []
    line = "o 1"
    with open('annotations.json', 'r') as file:
        finished_fragments = json.load(file)['finished']
    for fragment in finished_fragments:
        fragment_photo = read_image(f"{fragment}.jpeg")
        height, width = fragment_photo.shape[1], fragment_photo.shape[2]
        fragment_cursor = annotations.find({"fragmentNumber":fragment},{"annotations.geometry": 1, "annotations.croppedSign.label":1})
        for doc in fragment_cursor:
            annotations = doc['annotations']

        for cropped_sign in annotations:
            if cropped_sign['croppedSign']['label'] == "":
                continue
            if line == cropped_sign['croppedSign']['label']:
                x, y, w, h = cropped_sign['geometry']['x'], cropped_sign['geometry']['y'], \
                cropped_sign['geometry']['width'], cropped_sign['geometry']['height']
                xs.append(x), ys.append(y), ws.append(w), hs.append(h)
            else:
                line = cropped_sign['croppedSign']['label']
                x, y, w, h = line_boundaries(xs, ys, ws, hs)
                xs, ys, ws, hs = [], [], [], []
                absolute_x = int(round(x / 100 * width))
                absolute_y = int(round(y / 100 * height))
                absolute_w = int(round(w / 100 * width))
                absolute_h = int(round(h / 100 * height))
                bbox = [absolute_x, absolute_y, absolute_x + absolute_w, absolute_y + absolute_h]
                bbox = torch.tensor(bbox, dtype=torch.int)
                bbox = bbox.unsqueeze(0)
                img = draw_bounding_boxes(fragment_photo, bbox, width=10, colors=(255,255,0))
                img = ToPILImage()(img)
                img.show()
        break

  