from numpy import absolute
from pymongo import MongoClient
import tqdm
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
import requests
BASE_URL = "https://www.ebl.lmu.de/api/fragments"

def download_image(url, filename):
    respone = requests.get(url, stream=True)
    respone.raw.decode_content = True
    with open(filename, "wb") as outfile:
        outfile.write(respone.content)



def line_boundaries(xs, ys, ws, hs):
    x = min(xs)
    y = min(ys)
    w = sum(ws)
    h = max(hs)
    return x, y, w, h

def calculate_center(x, y, w, h):
    return (x + w / 2, y + h / 2)

def get_annotations(annotations, fragment):
        result = []
        fragment_cursor = annotations.find({"fragmentNumber":fragment},{"annotations.geometry": 1, "annotations.croppedSign.label":1})
        for doc in fragment_cursor:
            annotations = doc['annotations']
        for cropped_sign in annotations:
            x, y, w, h = cropped_sign['geometry']['x'], cropped_sign['geometry']['y'], \
            cropped_sign['geometry']['width'], cropped_sign['geometry']['height']
            result.append((x,y, w, h))
        return result

def calculate_union_bbox(signs, height, width, shrink_factor=0.5, max_height = 3):
    # Initialize min and max coordinates to the first sign's coordinates
    min_x = signs[0][0]
    max_x = signs[0][0] + signs[0][2]
    min_y = signs[0][1]
    max_y = signs[0][1] + signs[0][3]

    # Iterate through each sign and adjust min and max coordinates
    for (x, y, w, h) in signs:
        min_x = min(min_x, x)
        max_x = max(max_x, x + w)
        min_y = min(min_y, y)
        max_y = max(max_y, y + h)

    # Calculate the union bounding box
    union_x = min_x + shrink_factor
    union_y = min_y + shrink_factor
    union_w = max_x - min_x - (2 * shrink_factor)
    union_h = max_y - min_y - (2 * shrink_factor)
    if max_height is not None and union_h > max_height:
        return None
    union_w = max(0, union_w)
    union_h = max(0, union_h)

    
    absolute_x = int(round(union_x / 100 * width))
    absolute_y = int(round(union_y / 100 * height))
    absolute_w = int(round(union_w / 100 * width))
    absolute_h = int(round(union_h / 100 * height))
    bbox = [absolute_x, absolute_y, absolute_x + absolute_w, absolute_y + absolute_h]
    bbox = torch.tensor(bbox, dtype=torch.int)
    bbox = bbox.unsqueeze(0)
    return bbox

            


def group_signs_by_lines(signs, threshold):
    lines = []
    current_line = [signs[0]]

    for i in range(1, len(signs)):
        prev_center = calculate_center(*current_line[-1])
        curr_center = calculate_center(*signs[i])

        distance = abs(curr_center[0] - prev_center[0])  # horizontal distance
        print(distance)

        if distance < threshold:
            current_line.append(signs[i])
        else:
            lines.append(current_line)
            current_line = [signs[i]]

    if current_line:
        lines.append(current_line)

    return lines


if __name__ == '__main__':
    client  = MongoClient(fragments.CONNECTION)
    database = client["ebl"]
    images_id = 0
    annotations_id = 0
    fragmentarium = database["fragments"]
    annotations = database["annotations"]
    images = []
    bbox_annotations = []
    categories = [{"id": 0, "name" : "line"}]   

    with open('annotations.json', 'r') as file:
        finished_fragments = json.load(file)['finished']
        for fragment in tqdm.tqdm(finished_fragments):
            images_id += 1
            print(fragment)
            url = f"{BASE_URL}/{fragment}/photo"
            download_image(url, f"data/{fragment}.jpeg")
            image = read_image(f"data/{fragment}.jpeg")
            filename = f'{fragment}.jpeg'
            fragment_photo = read_image(f"data/{fragment}.jpeg")
            height, width = fragment_photo.shape[1], fragment_photo.shape[2]
            image_instance = {"file_name": filename, "height": height, "width": width, "id": images_id}
            images.append(image_instance)
            signs = get_annotations(annotations, fragment)
            print(signs)
            lines = group_signs_by_lines(signs, 2)
            union_bboxes = [calculate_union_bbox(line, height, width) for line in lines]
            # Filter out None values from the list
            filtered_bboxes = [bbox for bbox in union_bboxes if bbox is not None]
            for bbox in filtered_bboxes:
                image = draw_bounding_boxes(image, bbox, colors = (255,0,0), width=5)

                annotations_id += 1
                bbox = bbox[0].tolist()
                bbox_instance = {"image_id": images_id, "category_id": 0, "bbox": bbox, "id": annotations_id}
                bbox_annotations.append(bbox_instance)
            img = ToPILImage()(image)
            img.show()
    coco_format = {
        "images": images,
        "annotations": bbox_annotations,
        "categories": categories
    }
    with open('line_annotations.json', 'w') as file:
        json.dump(coco_format, file, indent=4)


  