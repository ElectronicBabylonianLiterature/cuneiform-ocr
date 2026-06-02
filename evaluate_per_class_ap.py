"""
Per-class AP evaluation for the 173-class cuneiform sign detector.

Runs inference via mmdetection, builds proper label→category_id mapping
for the val set, then computes per-class AP using pycocotools.

Usage (from mmdetection dir):
  cd mmdetection
  python ../evaluate_per_class_ap.py
"""

import sys
import os
import json
import numpy as np
from collections import Counter

# Add mmdetection to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/mmdetection')

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import mmcv
from mmengine.config import Config
from mmengine.runner import Runner

# ---- Config ----------------------------------------------------------------
CONFIG_PATH = '../configs/detr.py'
CHECKPOINT  = os.path.expanduser('~/erc-work-data/retrained_models/detr-173/epoch_1000.pth')
VAL_ANN_FILE = 'data/coco/annotations/instances_val2017.json'
TRAIN_ANN_FILE = 'data/coco/annotations/instances_train2017.json'

classes3 = [
    'TU', '|U.GUD|', 'TUM', 'LA', 'TA', 'GAR', 'GAL', 'I', 'TI', 'LI',
    'ZA', 'A', 'DI', 'MI', 'RI', 'IŠ', 'BA', 'LU', 'TE', 'DA',
    '|GUD×KUR|', 'MA', 'E₂', 'DIŠ', 'MU', 'DU', 'ŠU₂', 'EN', 'KUL', 'SI',
    '|I.A|', 'HI', 'MUŠ₃', 'AN', 'NA', 'BAD', 'AMAR', 'UD', 'UnclearSign',
    '|HI×BAD|', '|UD×(U.U.U)|', 'AB', 'AK', 'LUGAL', 'DIN', 'KI',
    'DUN₃@g', 'KU₃', 'AŠ', 'IGI', 'U₂', 'ŠA₃', 'BI', 'GUR', 'ŠE', 'ZI',
    'GA', 'SILA₃', 'ŠID', '|SAL.TUG₂|', 'SU', 'KAK', 'MAŠ', 'TUR', 'ŠEŠ',
    'LU₂', 'IA₂', 'UR', 'KAL', '|ŠEŠ.KI|', 'ZU', 'ŠU', 'NE', 'IM', 'RA',
    '|U.U|', 'ZAG', '|DIŠ.DIŠ.DIŠ|', 'GA₂', 'IN', 'KIN', 'TAR', 'MAH',
    'LAL', 'KID', 'GABA', 'KA', 'RU', 'ŠA', '|HI×NUN|', 'ME', 'BU', 'NI',
    'IG', 'MES', 'PA', 'SAG', 'U', 'E', 'GUM', 'GIŠ', '|U.KA|', 'LUM',
    '|HI×AŠ₂|', 'HA', 'UŠ', '|U.U.U|', 'MIN', 'NAM', 'NU', 'AL', 'AB₂',
    'IB', 'UM', 'KU', 'SUR', 'MEŠ', 'TUG₂', 'TAG', 'DIM', 'BAL', 'IR',
    'ERIN₂', 'PAP', 'SA', '|PIRIG×ZA|', 'UB', 'URU', '|U.5(DIŠ)|', 'DAM',
    'GAR₃', '|EN.ZU|', 'ZE₂', 'AD', 'APIN', 'EL', 'PI', 'AŠ@z', 'DAR',
    'DUB', 'SAR', 'GUD', 'A₂', 'KUR', 'ARAD', '|IGI.DIB|', '6(DIŠ)',
    'AŠ₂', 'IL', 'HU', 'NUN', 'SAL', 'GI', 'EŠ₂', 'UN', 'TIL', 'NIM',
    'TAB', 'SUM', '|3×AN|', '|NINDA₂×ŠE|', 'MAŠ₂', 'GI₄', 'GAN', 'DIM₂',
    'GU', 'MAR', 'MUŠ', 'BAR', '|IGI.RI|', 'TUK', '|UD.DU|', '|LAGAB.LAGAB|'
]
assert len(classes3) == 173

# ---- Build label→category_id mapping --------------------------------------
print('Loading annotations...')
coco_gt = COCO(VAL_ANN_FILE)

# Map class name → COCO category id (for classes present in val)
name_to_catid = {c['name']: c['id'] for c in coco_gt.dataset['categories']}

# label_to_catid[i] = COCO category id for classes3[i], or None if not in val
label_to_catid = [name_to_catid.get(name, None) for name in classes3]

# Training frequencies
print('Loading training annotations...')
with open(TRAIN_ANN_FILE) as f:
    train_data = json.load(f)
train_cats = {c['id']: c['name'] for c in train_data['categories']}
train_counts = Counter(a['category_id'] for a in train_data['annotations'])
# Map by name
train_count_by_name = {train_cats[cid]: cnt for cid, cnt in train_counts.items()}

# ---- Run inference --------------------------------------------------------
print('Loading config and model...')
cfg = Config.fromfile(CONFIG_PATH)
cfg.load_from = CHECKPOINT
cfg.resume = False
# Disable evaluation in runner - we'll evaluate ourselves
cfg.test_evaluator = dict(type='CocoMetric', ann_file=VAL_ANN_FILE,
                          metric='bbox', format_only=False)

runner = Runner.from_cfg(cfg)
# We'll collect predictions manually
runner.load_or_resume()

import torch
from mmdet.utils import get_test_pipeline_cfg
from mmengine.dataset import Compose

# Run inference via test dataloader and collect raw predictions
print('Running inference...')
model = runner.model
model.eval()

all_preds = []  # list of dicts: {image_id, bboxes, scores, labels}

test_loader = runner.test_dataloader
with torch.no_grad():
    for batch in test_loader:
        data = model.data_preprocessor(batch, False)
        results = model(**data, mode='predict')
        for r in results:
            img_id = r.img_id
            if hasattr(r, 'pred_instances'):
                pi = r.pred_instances
                bboxes = pi.bboxes.cpu().numpy()  # xyxy
                scores = pi.scores.cpu().numpy()
                labels = pi.labels.cpu().numpy()
            else:
                bboxes = np.zeros((0, 4))
                scores = np.zeros(0)
                labels = np.zeros(0, dtype=int)
            all_preds.append({
                'image_id': img_id,
                'bboxes': bboxes,
                'scores': scores,
                'labels': labels,
            })

print(f'Inference done. {len(all_preds)} images processed.')

# ---- Convert to COCO results format ---------------------------------------
print('Building COCO results...')
coco_results = []
for pred in all_preds:
    for bbox, score, label in zip(pred['bboxes'], pred['scores'], pred['labels']):
        catid = label_to_catid[int(label)]
        if catid is None:
            continue  # class not in val set, skip
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        coco_results.append({
            'image_id': pred['image_id'],
            'category_id': catid,
            'bbox': [float(x1), float(y1), float(w), float(h)],
            'score': float(score),
        })

print(f'Total predictions (in val classes): {len(coco_results)}')

# ---- Evaluate with pycocotools --------------------------------------------
print('\n=== Per-class AP Evaluation ===\n')
if len(coco_results) == 0:
    print('No predictions to evaluate!')
    sys.exit(1)

coco_dt = coco_gt.loadRes(coco_results)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Per-class AP (AP@0.5:0.95)
# coco_eval.eval['precision'] shape: [T, R, K, A, M]
# T=10 iou thresholds, R=101 recall points, K=num_categories, A=4 areas, M=3 max dets
precision = coco_eval.eval['precision']  # shape [10, 101, K, 4, 3]
# AP@0.5:0.95 = mean over T=10 thresholds, R=101 points, for all areas (A=0), maxDets=100 (M=2)
ap_per_class = precision[:, :, :, 0, 2].mean(axis=(0, 1))  # shape [K]

# Map back to class names
cat_ids_ordered = coco_eval.params.catIds  # ordered list of category ids evaluated
catid_to_name = {c['id']: c['name'] for c in coco_gt.dataset['categories']}

results = []
for k, catid in enumerate(cat_ids_ordered):
    name = catid_to_name[catid]
    ap = float(ap_per_class[k])
    train_count = train_count_by_name.get(name, 0)
    results.append({'name': name, 'catid': catid, 'ap': ap, 'train_count': train_count})

# Sort by training frequency
results_by_freq = sorted(results, key=lambda x: x['train_count'], reverse=True)

print('\n--- Overall mAP ---')
valid_aps = [r['ap'] for r in results if r['ap'] >= 0]
print(f'mAP (over {len(valid_aps)} classes in val): {np.mean(valid_aps):.4f}')

print('\n--- Top 10 by Training Frequency ---')
print(f'{"Rank":<5} {"Class":<20} {"Train Count":>12} {"AP@0.5:0.95":>12}')
print('-' * 55)
for i, r in enumerate(results_by_freq[:10]):
    print(f'{i+1:<5} {r["name"]:<20} {r["train_count"]:>12} {r["ap"]:>12.4f}')

print('\n--- Bottom 10 by Training Frequency (excluding 0-count) ---')
with_training = [r for r in results_by_freq if r['train_count'] > 0]
bottom10 = with_training[-10:][::-1]
print(f'{"Rank":<5} {"Class":<20} {"Train Count":>12} {"AP@0.5:0.95":>12}')
print('-' * 55)
for i, r in enumerate(bottom10):
    print(f'{i+1:<5} {r["name"]:<20} {r["train_count"]:>12} {r["ap"]:>12.4f}')

print('\n--- Classes with 0 training samples (in val set) ---')
zero_train = [r for r in results if r['train_count'] == 0]
print(f'{len(zero_train)} classes in val have 0 training samples')
if zero_train:
    print([r['name'] for r in zero_train])

# Save full results
out_path = '../per_class_ap_results.json'
with open(out_path, 'w') as f:
    json.dump({
        'results_by_frequency': results_by_freq,
        'mAP': float(np.mean(valid_aps)),
    }, f, indent=2, ensure_ascii=False)
print(f'\nFull results saved to {out_path}')
