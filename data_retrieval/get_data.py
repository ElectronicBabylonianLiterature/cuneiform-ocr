from pymongo import MongoClient
import fragments
import json
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import crop
from torchvision.transforms import ToTensor
if __name__ == '__main__':
    client  = MongoClient(fragments.CONNECTION)
    database = client["ebl"]
    fragmentarium = database["fragments"]
    annotations = database["annotations"]
    with open('annotations.json', 'r') as file:
        finished_fragments = json.load(file)['finished']
    for fragment in finished_fragments:
        fragment_photo = Image.open(f"{fragment}.jpeg")
        width, height = fragment_photo.size
        fragment_cursor = annotations.find({"fragmentNumber":fragment},{"annotations.geometry": 1})
        for doc in fragment_cursor:
            annotations = doc['annotations']
        for cropped_sign in annotations:
            x, y, w, h = cropped_sign['geometry']['x'], cropped_sign['geometry']['y'], \
            cropped_sign['geometry']['width'], cropped_sign['geometry']['height']
            absolute_x = int(round(x / 100 * width))
            absolute_y = int(round(y / 100 * height))
            absolute_width = int(round(w / 100 * width))
            absolute_height = int(round(h / 100 * height))
            sign = crop(fragment_photo, absolute_y, absolute_x, absolute_height, absolute_width)
            sign.show()
        break

  