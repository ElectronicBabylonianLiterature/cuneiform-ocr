## Installation
- Use mmdetection for task (library is very buggy and in general would not recommend maybe port code to detectron2 someday)

- Pytorch Version 2.0.1 & Cuda
- folder mmdetection (which is https://github.com/open-mmlab/mmdetection at commit: ecac3a77becc63f23d9f6980b2a36f86acd00a8a) is already downloaded !
- Install dependencies for mmdet https://mmdetection.readthedocs.io/en/latest/get_started.html
 ```python
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
- mmdetection is installed as in Step 1.a) in the link above
```python
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```


## Check Installation with Coco2017 Dataset
- Optional: Download Coco2017 Dataset (python tools/misc/download_dataset.py --dataset-name coco2017, see https://mmdetection.readthedocs.io/en/latest/user_guides/useful_tools.html#dataset-download) can take several hours
- python3 mmdetection/tools/analysis_tools/browse_dataset.py configs/mask_rcnn.py
- python3 mmdetection/tools/train.py configs/mask_rcnn.py
- python3 mmdetection/tools/test.py configs/mask_rcnn.py checkpoints/mask_rcnn.pth

## Ebl Dataset
- place data-coco from https://github.com/ElectronicBabylonianLiterature/cuneiform-ocr-data in root and rename to data
- run run runConfiguration/test_recognition.py

## For using browse_dataset or mmdetection only on detection (not classification or detection + classification)
- Delete all classes in data/annotations/val2017.json rename to train2017.json
- create train2017 folder next to val2017 folder
- Create one category in json:
 "categories": [
        {
            "id": 0,
            "name": "null"
        }]

- add: 
metainfo = {
    'classes': ('null', ),
    'palette': [
        (220, 20, 60),
    ]
}
to config.
- Replace all category ids in train2017.json with 0
- Replace all `"segmentation": [],` in train2017.json with `"segmentation": [[1,1,2,2,3,3,4,4]],` run 
- runConfiguration/test_detection.py or run runConfiguration/browse_dataset.py


error occurs at detection:
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
  File "/home/yunus/PycharmProjects/cuneiform-ocr-3/.venv/lib/python3.10/site-packages/mmengine/registry/registry.py", line 548, in build
    return self.build_func(cfg, *args, **kwargs, registry=self)
  File "/home/yunus/PycharmProjects/cuneiform-ocr-3/.venv/lib/python3.10/site-packages/mmengine/registry/build_functions.py", line 144, in build_from_cfg
    raise type(e)(
ValueError: class `CocoDataset` in mmdet/datasets/coco.py: need at least one array to concatenate


Is because the category ids is not correct either edit the json or the config file





Error occurs at detection:
  File "/home/yunus/PycharmProjects/cuneiform-ocr-3/mmdetection/tools/analysis_tools/browse_dataset.py", line 89, in <module>
    main()
  File "/home/yunus/PycharmProjects/cuneiform-ocr-3/mmdetection/tools/analysis_tools/browse_dataset.py", line 56, in main
    for item in dataset:
  File "/home/yunus/PycharmProjects/cuneiform-ocr-3/.venv/lib/python3.10/site-packages/mmengine/dataset/base_dataset.py", line 413, in __getitem__
    data = self.prepare_data(idx)
  File "/home/yunus/PycharmProjects/cuneiform-ocr-3/.venv/lib/python3.10/site-packages/mmengine/dataset/base_dataset.py", line 797, in prepare_data
    return self.pipeline(data_info)
  File "/home/yunus/PycharmProjects/cuneiform-ocr-3/.venv/lib/python3.10/site-packages/mmengine/dataset/base_dataset.py", line 59, in __call__
    data = t(data)
  File "/home/yunus/PycharmProjects/cuneiform-ocr-3/.venv/lib/python3.10/site-packages/mmcv/transforms/base.py", line 12, in __call__
    return self.transform(results)
  File "/home/yunus/PycharmProjects/cuneiform-ocr-3/mmdetection/mmdet/datasets/transforms/loading.py", line 400, in transform
    self._load_masks(results)
  File "/home/yunus/PycharmProjects/cuneiform-ocr-3/mmdetection/mmdet/datasets/transforms/loading.py", line 375, in _load_masks
    gt_masks = self._process_masks(results)
  File "/home/yunus/PycharmProjects/cuneiform-ocr-3/mmdetection/mmdet/datasets/transforms/loading.py", line 337, in _process_masks
    gt_mask = instance['mask']
KeyError: 'mask'

replace "segmentation": [], with  "segmentation": [[1,1,2,2,3,3,4,4]],
