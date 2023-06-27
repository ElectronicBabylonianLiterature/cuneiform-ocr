## Installation
Use mmdetection for task (horrible library)

- Pytorch & Cuda
- folder mmdetection (which is https://github.com/open-mmlab/mmdetection at commit: ecac3a77becc63f23d9f6980b2a36f86acd00a8a)
- Install dependencies for mmdet https://mmdetection.readthedocs.io/en/latest/get_started.html
- mmdetection is installed as in Step 1.a) in the link above


## Check Installation
- Download Coco2017 Dataset (python tools/misc/download_dataset.py --dataset-name coco2017, see https://mmdetection.readthedocs.io/en/latest/user_guides/useful_tools.html#dataset-download) can take several hours
- python3 mmdetection/tools/analysis_tools/browse_dataset.py configs/mask_rcnn.py
- python3 mmdetection/tools/train.py configs/mask_rcnn.py
- python3 mmdetection/tools/test.py configs/mask_rcnn.py checkpoints/mask_rcnn.pth

## For using browse_dataset or mmdetection only on detection (not classification or detection + classification)
- Delete all classes in ...val2017.json rename to train2017.json, create train2017 folder next to val2017 folder
- Create one category in json:
 "categories": [
        {
            "id": 0,
            "name": "null"
        }

    ]

- add: 
metainfo = {
    'classes': ('null', ),
    'palette': [
        (220, 20, 60),
    ]
}
to config.
- Replace all category ids in train2017.json with 0 