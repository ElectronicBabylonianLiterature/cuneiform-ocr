import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from ast import literal_eval
from urllib.request import urlopen
from urllib.error import HTTPError
import json
selected_collection = 'test'

def get_images(fragments):
    for i, (si, s_rec) in enumerate(fragments.iterrows()):
        print(s_rec.tablet_CDLI)
    url_template = 'https://cdli.ucla.edu/dl/photo/{}.jpg'
    image_url = url_template.format(s_rec.tablet_CDLI)
    try:
        im = Image.open(urlopen(image_url))
        im.save('./data/CDLI_images/{}.jpg'.format(s_rec.tablet_CDLI))
    except HTTPError as e:
        if e.code == 500:
            print("Internal Server Error (500) occurred.")

def crop_segment_from_image(im, bb):
    """
    im: pil image
    bb: list of segment bounding box coordinates as xmin, ymin, xmax, ymax.
    return: cropped segment as pil image
    """
    return im.crop((bb[0], bb[1], bb[2], bb[3]))

line_df = pd.read_csv('./line_annotations_{}.csv'.format(selected_collection))
# read csv file
seg_df = pd.read_csv('./tablet_segments_{}.csv'.format(selected_collection))
# convert bbox to np array
seg_df.bbox = seg_df.bbox.apply(literal_eval).apply(np.array)
# remove unassigned segments
seg_df = seg_df[seg_df.assigned]
# show first few entries
fragments = seg_df
images = []
bbox_annotations = []
categories = [{"id": 0, "name" : "line"}]   
images_id = 0
annotations_id = 0
for i, (si, s_rec) in enumerate(fragments.iterrows()):
    try:
        path_to_image = './data/CDLI_images/{}.jpg'.format(s_rec.tablet_CDLI)
        pil_im = Image.open(path_to_image)
        tablet_seg = crop_segment_from_image(pil_im, s_rec.bbox)
        height, width = tablet_seg.height, tablet_seg.width

    # crop segment
        filename = './data/CDLI_images/segments/{}_{}.jpg'.format(s_rec.tablet_CDLI, s_rec.segm_idx)
        tablet_seg.save(filename)
        image_instance = {"file_name": filename.split('/')[-1], "height": height, "width": width, "id": images_id}
        images.append(image_instance)
        line_df_slice = line_df[line_df.segm_idx == s_rec.segm_idx]

    # plot line annotations
        grouped = line_df_slice.groupby('line_idx')
        for li, line_rec in grouped:
            top_left_x = line_rec['x'].min()
            top_left_y = line_rec['y'].min()
            bottom_right_x = line_rec['x'].max()
            bottom_right_y = line_rec['y'].max()
            bbox = [top_left_x, top_left_y, bottom_right_x - top_left_x, bottom_right_y - top_left_y]
            bbox_instance = {"image_id": images_id, "category_id": 0, "bbox": bbox, "id": annotations_id}
            bbox_annotations.append(bbox_instance)
            annotations_id += 1
        images_id += 1
    except FileNotFoundError:
        print("The file was not found.")
    # open tablet image
    
coco_format = {
        "images": images,
        "annotations": bbox_annotations,
        "categories": categories
    }
with open('CDLI_line_annotations_test.json', 'w') as file:
    json.dump(coco_format, file, indent=4)