#!/usr/bin/env python3
"""The repository's single-stage Deformable DETR, run tile by tile on whole tablet photographs.

    python tiled_inference_detr.py CONFIG.py WEIGHTS.safetensors classes173.json PHOTO_DIR scales.json 64 out.json [--tile 800]

Each photograph is resized so that its signs are SIGN_H pixels tall (per-photograph scale from scales.json, which
gives the scale for 64-pixel signs: write it with tiled_inference_yolo.py --write-scales, or compute it from the
median sign height), cut into TILE x TILE tiles (stride 3/4 tile), and every tile goes through the config's test
pipeline unchanged in size (a tile of 800 pixels or less passes Resize(1333, 800) with keep_ratio as it is). Up to
300 boxes are kept per tile (the model's queries). Boxes cut by an interior tile border are dropped (the neighbouring
tile sees them whole), then class-agnostic NMS at IoU 0.5 across tiles. Every box with a score of 0.05 or more is
written with its class, so the threshold can be chosen afterwards.

  CONFIG.py      this repository's DETR config (the 173-class model: configs/detr.py)
  WEIGHTS        a state dict as safetensors (detr173-ft1b-e9.safetensors, see README.md)
  scales.json    {photograph name without .jpg: scale at which its signs are 64 pixels tall}
  out.json       {name: [[x1, y1, x2, y2, score, class], ...]} in the photograph's own pixels

Needs this repository's mmdetection setup (see the main README) plus: safetensors, opencv-python, torchvision.
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import torchvision
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config
from safetensors.torch import load_file

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('config'); ap.add_argument('weights'); ap.add_argument('classes'); ap.add_argument('photos')
ap.add_argument('scales'); ap.add_argument('sign_h', type=float); ap.add_argument('out')
ap.add_argument('--tile', type=int, default=800); ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--device', default='cuda:0')
a = ap.parse_args()

cfg = Config.fromfile(a.config)
cfg.model.backbone.init_cfg = None
cfg.model.test_cfg = dict(max_per_img=300)
pipe = [s for s in cfg.test_dataloader.dataset.pipeline if s['type'] not in ('LoadAnnotations', 'mmdet.LoadAnnotations')]
cfg.test_dataloader.dataset.pipeline = pipe
cfg.test_pipeline = pipe
model = init_detector(cfg, checkpoint=None, device=a.device)
sd = load_file(a.weights)
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not missing and not unexpected, f'the weights do not fit the config: missing {missing[:5]}, unexpected {unexpected[:5]}'
model.eval()
CLASSES = json.load(open(a.classes))
SCALES = json.load(open(a.scales))
out = json.load(open(a.out)) if os.path.exists(a.out) else {}
T = a.tile; stride = T * 3 // 4; margin = 4

for name, s64 in sorted(SCALES.items()):
    if name in out:
        continue
    im = cv2.imread(os.path.join(a.photos, name + '.jpg'))
    if im is None:
        print(name, 'not readable'); continue
    sc = s64 * a.sign_h / 64.0
    work = cv2.resize(im, (max(32, round(im.shape[1] * sc)), max(32, round(im.shape[0] * sc))), interpolation=cv2.INTER_AREA)
    H, W = work.shape[:2]
    if H < T or W < T:
        canvas = np.zeros((max(H, T), max(W, T), 3), np.uint8); canvas[:H, :W] = work; work = canvas
    H2, W2 = work.shape[:2]
    xs = list(range(0, W2 - T + 1, stride)); ys = list(range(0, H2 - T + 1, stride))
    if (W2 - T) % stride: xs.append(W2 - T)
    if (H2 - T) % stride: ys.append(H2 - T)
    origins = [(x, y) for y in ys for x in xs]
    B, S, L = [], [], []
    for i in range(0, len(origins), a.batch):
        chunk = origins[i:i + a.batch]
        results = inference_detector(model, [np.ascontiguousarray(work[y:y + T, x:x + T]) for x, y in chunk])
        for (x0, y0), r in zip(chunk, results):
            p = r.pred_instances
            b = p.bboxes.cpu().numpy(); s = p.scores.cpu().numpy(); l = p.labels.cpu().numpy()
            k = s >= 0.05
            for (x1, y1, x2, y2), ss, ll in zip(b[k], s[k], l[k]):
                if (x1 < margin and x0 > 0) or (y1 < margin and y0 > 0) or (x2 > T - margin and x0 + T < W2) or (y2 > T - margin and y0 + T < H2):
                    continue
                B.append([x1 + x0, y1 + y0, x2 + x0, y2 + y0]); S.append(float(ss)); L.append(int(ll))
    B = np.array(B, float).reshape(-1, 4); S = np.array(S); L = np.array(L, int)
    B[:, [0, 2]] = B[:, [0, 2]].clip(0, W); B[:, [1, 3]] = B[:, [1, 3]].clip(0, H)
    keep = torchvision.ops.nms(torch.tensor(B, dtype=torch.float32), torch.tensor(S, dtype=torch.float32), 0.5).numpy() if len(B) else []
    out[name] = [[round(float(v) / sc, 1) for v in B[i]] + [round(float(S[i]), 4), CLASSES[int(L[i])] if int(L[i]) < len(CLASSES) else str(int(L[i]))]
                 for i in keep]
    print(name, 'tiles', len(origins), 'boxes', len(out[name]), flush=True)
    json.dump(out, open(a.out + '.part', 'w')); os.replace(a.out + '.part', a.out)
print('done')
