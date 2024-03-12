# Cuneiform OCR Data Preprocessing for Ebl Project (https://www.ebl.lmu.de/, https://github.com/ElectronicBabylonianLiterature
Data+Code is part of Paper **Sign Detection for Cuneiform Tablets from Yunus Cobanoglu, Luis Sáenz, Ilya Khait, Enrique Jiménez** please contact us for access to data on Zenodoo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10693601.svg)](https://doi.org/10.5281/zenodo.10693601) and paper as it is under currently under review.

# General Information

The Code and Data ([![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10693601.svg)](https://doi.org/10.5281/zenodo.10693601))  is used together with the following Github Repositories and deployed partially with the [eBL](https://www.ebl.lmu.de/) platform and part of the [eBL Organization on Github](https://github.com/ElectronicBabylonianLiterature).

## Github Repositories

### [cuneiform-ocr-data](https://github.com/ElectronicBabylonianLiterature/cuneiform-ocr-data)

requires `raw-data` and `processed-data` and will convert the data in a ready for training format which is also available as `ready-for-training` here.

ready-for-training contains:
- coco-two-stage
- coco-recognition
- icdar2015
- icdar2015-deebscripe
- classification.tar

### [cuneiform-ocr-classification-detection](https://github.com/ElectronicBabylonianLiterature/cuneiform-ocr-classification-detection)

- requires  either `icdar2015` or `icdar2015-deebscribe` to train detection model or classification.tar to train classification model
- requires checkpoints fcenet and efficient-net
- may require pretrained weights for fcenet and efficient-net which can be download according to [mmocr](https://github.com/open-mmlab/mmocr) or [mmpretrain](https://github.com/open-mmlab/mmpretrain) documentation


### [cuneiform-ocr](https://github.com/ElectronicBabylonianLiterature/cuneiform-ocr)

- requires `coco-two-stage` for evaluating two-stage model or `coco-recognition` to train single-stage model (coco-recognition has to be placed in as 'mmdetection/data' and root has to be mmdetection to run 'tools/test.py' for the one-stage model. Every else is similar as in mmdetection please consult there docs.
- requires checkpoint detr for single-stage model

### [ebl-ai-api](https://github.com/ElectronicBabylonianLiterature/ebl-ai-api)

- requires checkpoint `ebl-ai-api` for only detection model (fcenet) and deployement in [eBL](https://www.ebl.lmu.de/)


## Installation (for errors during installation see bottom of README.md)
- Use mmdetection for task (library is very buggy and in general would not recommend maybe port code to detectron2 someday)

- Pytorch Version 2.0.1 & Cuda
- folder mmdetection (which is https://github.com/open-mmlab/mmdetection at commit: ecac3a77becc63f23d9f6980b2a36f86acd00a8a) is already downloaded !
- Install dependencies for mmdet https://mmdetection.readthedocs.io/en/latest/get_started.html
 ```python
pip install -U openmim
mim install mmengine
`mim install "mmcv>=2.0.0"`
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

## Single Stage Model and Two-Stage Model Setups (Important)
For Two-Stage: in configs file (like `configs/fcenet.py`) `metainfo` is the field where the classes are specified. The final model output is a number which correspond to `sorted(classes)`. Metainfo will be copied over to `cuneiform-ocr/mmdetection/mmdet/datasets/coco.py` classes and palette (palette are the colored that the bounding boxes are displayed in).Classes are also specified in config/efficient_net.py

For Single-Stage: Theoretically single stage setup for classes should work the same but it doesn't which seems like a bug. So metainfo from `mmdetection/custom_configs/detr.py` will not be copied over so you have to manually copy the classes to `cuneiform-ocr/mmdetection/mmdet/datasets/coco.py`. The classes are determined in the train_instances.json and test_instances.json (at the bottom) from the data folder (data-recognition).

## Data and Checkpoints Two-Stage Model
Data [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10693501.svg)](https://doi.org/10.5281/zenodo.10693501) download ready-for-training.tar.gz

To evaluate two-stage model place coco-two-stage/data to cuneiform-ocr/data

Checkpoints of efficient-net and fcenet have to downloaded and placed into checkpoints. cuneiform-ocr/config contains now similar configs as the ones downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10693501.svg)](https://doi.org/10.5281/zenodo.10693501) but no exactly the same. The config are adapted to work with mmdetection library. Originally fcenet is trained with mmocr and efficient-net with mmpretrain (previously mmcls). The configs in the zenodoo repository are configs from precisely those configs (mmocr/mmpretrain). The corresponding configs in  cuneiform-ocr/config are variants of those configs compatible with mmdetection. In case you train a new two-stage model and you want to evaluate it you have to make sure to copy paste the parts to cuneiform-ocr/config from the configs from mmocr/mmpretrain to make sure you can evaluate it. Subtle bugs (for example evaluation doesn't work because there is no data passed to the model) can happen if there are some errors. The main part in the modified config in cuneiform-ocr/config is the "model" section. Copying that from the mmocr/mmpretrain configs should be enough to that it can be used for two-stage evaluation in mmdetection. Also note that many models contains a threshold score like FCENET which has to be manually set to a value which has the maximum F1-Score to maximize performance.

- To run evaluation on two-stage run cuneiform-ocr/test.py and pass configs + checkpoints (fcenet config and checkpoint as cli and efficient-net config and checkpoint in code)
- For debugging, useful to use cuneiform-ocr/mmdetection/tools/analysis_tools/browse_dataset.py to display images + annotations after training pipeline transformation have been applied
- recognition_model.py is the model which combines the two stage and performs line detection based on Ransac Regression Model
- inference.py uses the two-stage model and ebl-ai public api to download images and create .txt output using the OCR Model and Line Detection
  
## Data and Checkpoints Single-Stage Model
- To train single-stage detection place coco-recognition/data into cuneiform-ocr/mmdetection/data
- Download detr-2024 config and checkpoint
- Run cuneiform-ocr/mmdetection/tools/test.py

Training and testing the single-stage model is just an application of mmdetection library. (One could also create a new repository just using mmdetection (make sure it is same version as here, please do not try to use newer version as library is very buggy and not likely not backward compatible) to train and test the single-stage model)

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

## Bugs 
error occurs at detection:
```
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
  File "/home/yunus/PycharmProjects/cuneiform-ocr-3/.venv/lib/python3.10/site-packages/mmengine/registry/registry.py", line 548, in build
    return self.build_func(cfg, *args, **kwargs, registry=self)
  File "/home/yunus/PycharmProjects/cuneiform-ocr-3/.venv/lib/python3.10/site-packages/mmengine/registry/build_functions.py", line 144, in build_from_cfg
    raise type(e)(
ValueError: class `CocoDataset` in mmdet/datasets/coco.py: need at least one array to concatenate
```

Is because the category ids is not correct either edit the json or the config file





Error occurs at detection or browse_dataset.py:
```
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
```
replace "segmentation": [], with  "segmentation": [[1,1,2,2,3,3,4,4]],

```
Traceback (most recent call last):
  File "/home/yunus/PycharmProjects/cuneiform-text-detection/cuneiform-ocr/test.py", line 27, in <module>
    from mmdet.engine.hooks.utils import trigger_visualization_hook
  File "/home/yunus/PycharmProjects/cuneiform-text-detection/cuneiform-ocr/mmdetection/mmdet/__init__.py", line 17, in <module>
    and mmcv_version < digit_version(mmcv_maximum_version)), \
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: MMCV==2.1.0 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.1.0.
```

Comment Out Assertion Statement
