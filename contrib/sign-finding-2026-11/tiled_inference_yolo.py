#!/usr/bin/env python3
"""Tiled sign finding with a YOLO11 detector (one class: "sign") on whole tablet photographs.

    python tiled_inference_yolo.py sign-detector-u1.pt PHOTO.jpg [PHOTO.jpg ...] --out boxes.json
        [--sign-height PX | --scale S] [--conf 0.25] [--write-scales scales.json] [--device cpu]

The detector was trained on 640 x 640 tiles in which a sign is about 64 pixels tall, so a photograph is first
resized until its signs have that height, then cut into overlapping tiles (stride 448). Boxes cut by an interior
tile border are dropped (the neighbouring tile sees them whole) and the rest are merged across tiles.

  --sign-height PX   median height of a sign in the photograph, in pixels (the photograph is scaled by 64 / PX)
  --scale S          the scale itself
  neither            the scale is searched: the photograph is tried at several sizes and the one at which the
                     confident boxes are closest to 64 pixels tall is kept (needs a dozen readable signs)
  --write-scales     also write {name: scale} for tiled_inference_detr.py

Output: {photograph name: [[x1, y1, x2, y2, score], ...]} in the photograph's own pixels.
Needs: ultralytics, numpy, pillow.
"""
import argparse
import json
import math
import os

import numpy as np
from PIL import Image
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None       # museum photographs can be larger than Pillow's default limit (179 megapixels)
TILE, STRIDE, SIGN_H = 640, 448, 64.0
LONG_SIDES = (400, 560, 800, 1120, 1600, 2240, 3136, 4390)     # sizes tried by the scale search


def tile_origins(w, h):
    xs = list(range(0, max(1, w - TILE + 1), STRIDE))
    ys = list(range(0, max(1, h - TILE + 1), STRIDE))
    if w > TILE and (w - TILE) % STRIDE:
        xs.append(w - TILE)
    if h > TILE and (h - TILE) % STRIDE:
        ys.append(h - TILE)
    return [(x, y) for y in ys for x in xs]


def nms(boxes, scores, iou_thr=0.5, contain_thr=0.75):
    order = np.argsort(-scores)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while len(order):
        i, rest = order[0], order[1:]
        keep.append(i)
        if not len(rest):
            break
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0]); y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2]); y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        contain = inter / (np.minimum(area[i], area[rest]) + 1e-9)
        order = rest[(iou <= iou_thr) & (contain <= contain_thr)]
    return keep


def detect(model, image, device, conf=0.05):
    """image: PIL RGB already at the working scale -> (boxes N x 4, scores N) in its pixels."""
    w, h = image.size
    if w < TILE or h < TILE:
        canvas = Image.new('RGB', (max(w, TILE), max(h, TILE)))
        canvas.paste(image, (0, 0))
        image = canvas
    W, H = image.size
    arr = np.asarray(image)
    origins = tile_origins(W, H)
    boxes, scores = [], []
    for i in range(0, len(origins), 16):
        chunk = origins[i:i + 16]
        batch = [np.ascontiguousarray(arr[y:y + TILE, x:x + TILE, ::-1]) for x, y in chunk]      # BGR for ultralytics
        for (x0, y0), r in zip(chunk, model.predict(batch, imgsz=TILE, conf=conf, iou=0.6, device=device, verbose=False, max_det=600)):
            for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                m = 3
                if (x1 < m and x0 > 0) or (y1 < m and y0 > 0) or (x2 > TILE - m and x0 + TILE < W) or (y2 > TILE - m and y0 + TILE < H):
                    continue
                boxes.append([x1 + x0, y1 + y0, x2 + x0, y2 + y0]); scores.append(float(s))
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0)
    boxes, scores = np.array(boxes), np.array(scores)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w); boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    keep = nms(boxes, scores)
    return boxes[keep], scores[keep]


def at_scale(model, photo, scale, device, conf=0.05):
    work = photo.resize((max(32, round(photo.width * scale)), max(32, round(photo.height * scale))), Image.BILINEAR)
    boxes, scores = detect(model, work, device, conf)
    return boxes / scale, scores


def search_scale(model, photo, device):
    """The scale at which the confident boxes are closest to SIGN_H pixels tall (None when too few signs are found)."""
    best = None
    for side in LONG_SIDES:
        s = side / max(photo.size)
        if s > 4:
            continue
        boxes, scores = at_scale(model, photo, s, device, conf=0.25)
        if len(boxes) < 12:
            continue
        height = float(np.median(boxes[:, 3] - boxes[:, 1])) * s
        miss = abs(math.log(height / SIGN_H))
        if best is None or miss < best[0]:
            best = (miss, s * SIGN_H / height)          # one step of correction toward the trained height
    return best[1] if best else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('weights'); ap.add_argument('photos', nargs='+'); ap.add_argument('--out', required=True)
    ap.add_argument('--sign-height', type=float); ap.add_argument('--scale', type=float)
    ap.add_argument('--conf', type=float, default=0.25); ap.add_argument('--write-scales'); ap.add_argument('--device', default=None)
    a = ap.parse_args()
    model = YOLO(a.weights)
    out, scales = {}, {}
    for path in a.photos:
        name = os.path.splitext(os.path.basename(path))[0]
        photo = Image.open(path).convert('RGB')
        scale = a.scale or (SIGN_H / a.sign_height if a.sign_height else search_scale(model, photo, a.device))
        if scale is None:
            print(f'{name}: too few signs found to choose a scale; give --sign-height'); out[name] = []; continue
        boxes, scores = at_scale(model, photo, scale, a.device)
        k = scores >= a.conf
        out[name] = [[round(float(v), 1) for v in b] + [round(float(s), 4)] for b, s in zip(boxes[k], scores[k])]
        scales[name] = round(float(scale), 4)
        print(f'{name}: scale {scale:.3f}, {int(k.sum())} boxes at {a.conf}')
    json.dump(out, open(a.out, 'w'))
    if a.write_scales:
        json.dump(scales, open(a.write_scales, 'w'), indent=1)


if __name__ == '__main__':
    main()
