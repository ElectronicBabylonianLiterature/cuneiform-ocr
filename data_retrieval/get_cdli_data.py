import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from ast import literal_eval
from urllib.request import urlopen
from urllib.error import HTTPError
selected_collection = 'train'

def crop_segment_from_image(im, bb):
    """
    im: pil image
    bb: list of segment bounding box coordinates as xmin, ymin, xmax, ymax.
    return: cropped segment as pil image
    """
    return im.crop((bb[0], bb[1], bb[2], bb[3]))

line_df = pd.read_csv('./line_annotations_{}.csv'.format(selected_collection))
print(line_df.head())
# read csv file
seg_df = pd.read_csv('./tablet_segments_{}.csv'.format(selected_collection))
# convert bbox to np array
seg_df.bbox = seg_df.bbox.apply(literal_eval).apply(np.array)
# remove unassigned segments
seg_df = seg_df[seg_df.assigned]
# show first few entries
fragments = seg_df[['tablet_CDLI','segm_idx', 'bbox']]
print(len(fragments))

fig, ax = plt.subplots(38,1, figsize=(18, 22))
for i, (si, s_rec) in enumerate(fragments.iterrows()):
    url_template = 'https://cdli.ucla.edu/dl/photo/{}.jpg'
    image_url = url_template.format(s_rec.tablet_CDLI)
    try:
        im = Image.open(urlopen(image_url))
        im.save('./data/CDLI_images/{}.jpg'.format(s_rec.tablet_CDLI))
    except HTTPError as e:
        if e.code == 500:
            print("Internal Server Error (500) occurred.")
    try:
        path_to_image = './data/CDLI_images/{}.jpg'.format(s_rec.tablet_CDLI)
        pil_im = Image.open(path_to_image)
        
    except FileNotFoundError:
        print("The file was not found.")
    # open tablet image
    tablet_seg = crop_segment_from_image(pil_im, s_rec.bbox)
    # crop segment

    # select line annotations
    line_df_slice = line_df[line_df.segm_idx == s_rec.segm_idx]

    # plot line annotations
    grouped = line_df_slice.groupby('line_idx')
    for li, line_rec in grouped:
        # plot single line as piece-wise linear function
        ax[i].plot(line_rec.x.values, line_rec.y.values, linewidth=5)
        # annotated line with line index
        ax[i].text(line_rec.x.values[0], line_rec.y.values[0], 
                   '{}'.format(line_rec.line_idx.values[0]),
                   bbox=dict(facecolor='blue', alpha=0.5), fontsize=8, color='white')  
    # plot image
    ax[i].imshow(np.asarray(tablet_seg), cmap='gray')
    ax[i].set_title("{}".format(s_rec.tablet_CDLI))
plt.show()