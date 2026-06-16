import base64
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image

from .sign import Sign, SignResolver
from .box import Box, Boxes
from .tablet import Tablet


def _safe_cache_part(value: Optional[str]) -> str:
    if not value:
        return "none"
    text = str(value)
    readable = re.sub(r"[^\w.@+-]+", "_", text, flags=re.UNICODE).strip("_")
    readable = readable[:80] or "value"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{digest}"


def _decode_base64_gray(image: str) -> np.ndarray:
    if not image:
        raise ValueError("empty image payload")
    if image.startswith("data:image/"):
        image = image.split(",", 1)[1]
    raw = base64.b64decode(image)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("L"))


class CanonicalSignSource:
    def for_period(self, period: str) -> "CanonicalSignSource":
        return self

    def is_ready(self) -> bool:
        return bool(self.list_sign_names())

    def get_image(self, sign: Sign) -> Optional[np.ndarray]:
        raise NotImplementedError

    def get_id(self, sign: Sign) -> Optional[str]:
        raise NotImplementedError

    def list_sign_names(self) -> List[str]:
        raise NotImplementedError

    def cache_namespace(self) -> str:
        return type(self).__name__

    def cache_file_stem(self, sign: Sign) -> Optional[str]:
        sid = self.get_id(sign)
        return _safe_cache_part(sid) if sid else None

    def describe(self) -> str:
        return type(self).__name__


class EBLMongoCanonicalSource(CanonicalSignSource):
    def __init__(
        self,
        mongodb_uri: Optional[str],
        period: Optional[str] = None,
        db_name: str = "ebl",
        form: str = "canonical1",
        require_centroid: bool = True,
    ):
        self.mongodb_uri = mongodb_uri
        self.period = period
        self.db_name = db_name
        self.form = form
        self.require_centroid = require_centroid
        self._images_by_sign: Dict[str, np.ndarray] = {}
        if period is not None:
            self._load()

    def for_period(self, period: str) -> "EBLMongoCanonicalSource":
        if period == self.period:
            return self
        return EBLMongoCanonicalSource(
            mongodb_uri=self.mongodb_uri,
            period=period,
            db_name=self.db_name,
            form=self.form,
            require_centroid=self.require_centroid,
        )

    def get_image(self, sign: Sign) -> Optional[np.ndarray]:
        return self._images_by_sign.get(sign.name)

    def get_id(self, sign: Sign) -> Optional[str]:
        if sign.name not in self._images_by_sign:
            return None
        return f"ebl/{self.period or 'period-missing'}/{self.form}/{sign.name}"

    def list_sign_names(self) -> List[str]:
        return sorted(self._images_by_sign)

    def cache_namespace(self) -> str:
        return f"ebl-mongo:{self.period or 'period-missing'}:{self.form}"

    def cache_file_stem(self, sign: Sign) -> Optional[str]:
        if sign.name not in self._images_by_sign:
            return None
        return "__".join(
            _safe_cache_part(part)
            for part in (sign.name, self.period or "period-missing", self.form)
        )

    def describe(self) -> str:
        return (
            f"eBL Mongo source period={self.period!r}, form={self.form!r}, "
            f"canonical_images={len(self._images_by_sign)}"
        )

    def _load(self) -> None:
        if not self.mongodb_uri or self.mongodb_uri == "YOUR_MONGODB_URI":
            raise ValueError("MONGODB_URI is not configured")

        from pymongo import MongoClient

        client = MongoClient(self.mongodb_uri)
        try:
            docs = client[self.db_name]["annotations"].aggregate(
                self._pipeline(), allowDiskUse=True
            )
            for item in docs:
                sign_name = item.get("signName")
                if not sign_name:
                    raise ValueError(f"annotation without signName: {item}")
                if sign_name not in self._images_by_sign:
                    self._images_by_sign[sign_name] = _decode_base64_gray(
                        item.get("image", "")
                    )
        finally:
            client.close()

        if not self._images_by_sign:
            period_msg = f" for period {self.period!r}" if self.period else ""
            raise RuntimeError(f"no {self.form!r} canonical images found{period_msg}")

    def _pipeline(self) -> List[Dict[str, Any]]:
        annotation_conditions: List[Dict[str, Any]] = [
            {"$eq": ["$$annotation.data.type", "HasSign"]},
            {"$eq": ["$$annotation.pcaClustering.form", self.form]},
            {
                "$ne": [
                    {"$ifNull": ["$$annotation.croppedSign.imageId", None]},
                    None,
                ]
            },
        ]
        if self.require_centroid:
            annotation_conditions.append(
                {"$eq": ["$$annotation.pcaClustering.isCentroid", True]}
            )

        pipeline: List[Dict[str, Any]] = [
            {
                "$match": {
                    "annotations.pcaClustering.form": self.form,
                    "annotations.croppedSign.imageId": {"$exists": True},
                }
            },
            {
                "$lookup": {
                    "from": "fragments",
                    "localField": "fragmentNumber",
                    "foreignField": "_id",
                    "as": "fragment",
                }
            },
            {"$unwind": "$fragment"},
        ]
        if self.period:
            pipeline.append({"$match": {"fragment.script.period": self.period}})

        pipeline.extend(
            [
                {
                    "$project": {
                        "fragmentNumber": 1,
                        "annotations": {
                            "$filter": {
                                "input": "$annotations",
                                "as": "annotation",
                                "cond": {"$and": annotation_conditions},
                            }
                        },
                        "date": "$fragment.date",
                        "period": "$fragment.script.period",
                        "script": "$fragment.script",
                        "provenance": "$fragment.archaeology.site",
                    }
                },
                {"$unwind": "$annotations"},
                {
                    "$lookup": {
                        "from": "cropped_sign_images",
                        "localField": "annotations.croppedSign.imageId",
                        "foreignField": "_id",
                        "as": "imageDoc",
                    }
                },
                {"$unwind": "$imageDoc"},
                {
                    "$project": {
                        "_id": 0,
                        "fragmentNumber": 1,
                        "signName": "$annotations.data.signName",
                        "image": "$imageDoc.image",
                        "script": 1,
                        "period": 1,
                        "label": "$annotations.croppedSign.label",
                        "date": 1,
                        "provenance": 1,
                        "annotationId": "$annotations.data.id",
                        "pcaClustering": "$annotations.pcaClustering",
                    }
                },
                {
                    "$sort": {
                        "signName": 1,
                        "pcaClustering.isMain": -1,
                        "pcaClustering.clusterSize": -1,
                        "pcaClustering.clusterRank": 1,
                    }
                },
            ]
        )
        return pipeline


class SignAPIResolver:
    SIGNS_API_URL = "https://www.ebl.lmu.de/api/signs"

    def __init__(self, cache_file: str = None):
        if cache_file is None:
            cache_file = os.path.join(os.path.dirname(__file__), '.sign_api_cache.json')
        self._cache_file = cache_file
        self._cache = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    def _save_cache(self):
        with open(self._cache_file, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False)

    def _cache_key(self, value: str, sub_index: int) -> str:
        return f"{value}|{sub_index}"

    def resolve(self, value: str, sub_index: int) -> Optional[str]:
        key = self._cache_key(value.lower(), sub_index)
        if key in self._cache:
            return self._cache[key]

        result = self._query_api(value.lower(), sub_index)
        self._cache[key] = result
        self._save_cache()
        return result

    def _query_api(self, value: str, sub_index: int) -> Optional[str]:
        try:
            resp = requests.get(
                self.SIGNS_API_URL,
                params={'value': value, 'subIndex': sub_index},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list) and len(data) > 0:
                    return data[0].get('name', None)
        except requests.RequestException:
            pass
        return None


class LocalDataSource:
    def __init__(self, annotations_dir: str):
        self.annotations_dir = Path(annotations_dir)
        self.imgs_path = self.annotations_dir / "imgs"
        self.annotations_path = self.annotations_dir / "annotations"

        if not self.imgs_path.exists():
            raise ValueError(f"Images directory not found: {self.imgs_path}")
        if not self.annotations_path.exists():
            raise ValueError(f"Annotations directory not found: {self.annotations_path}")

    def get_available_fragments(self) -> List[str]:
        fragments = []
        for img_file in os.listdir(self.imgs_path):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                fragment_id = os.path.splitext(img_file)[0]
                gt_file = self.annotations_path / f"gt_{fragment_id}.txt"
                if gt_file.exists():
                    fragments.append(fragment_id)
        return fragments

    def load_image(self, fragment_id: str) -> Optional[cv2.Mat]:
        possible_names = [
            f"{fragment_id}.jpg",
            f"{fragment_id}.jpeg",
            f"{fragment_id}.png",
        ]
        for name in possible_names:
            filepath = self.imgs_path / name
            if filepath.exists():
                return cv2.imread(str(filepath))
        return None

    def load_annotation(self, fragment_id: str, tablet: Tablet) -> Optional[Boxes]:
        gt_file = self.annotations_path / f"gt_{fragment_id}.txt"
        if not gt_file.exists():
            return None

        boxes = Boxes(tablet=tablet)
        with open(gt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    x, y, w, h = (
                        int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    )
                    sign_name = parts[4]
                    sign = SignResolver.resolve(sign_name, expected_type='SIGN')
                    bbox = Box(
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + w),
                        y2=float(y + h),
                        score=1.0,
                        sign=sign,
                        tablet=tablet,
                    )
                    boxes.append(bbox)

        return boxes if boxes else None


class LocalTestDataSource:
    ANNOTATION_FILE = "annotations/instances_val2017.json"
    IMAGES_DIR = "val2017"

    def __init__(self, coco_dir: str):
        self.coco_dir = Path(coco_dir)
        self._data: Optional[dict] = None
        self._image_by_stem: dict = {}
        self._annotations_by_image: dict = {}
        self._category_names: dict = {}

    def _ensure_loaded(self):
        if self._data is not None:
            return

        ann_path = self.coco_dir / self.ANNOTATION_FILE
        with open(ann_path, 'r', encoding='utf-8') as f:
            self._data = json.load(f)

        self._category_names = {c['id']: c['name'] for c in self._data['categories']}

        for img in self._data['images']:
            stem = img['file_name'].rsplit('.', 1)[0]
            self._image_by_stem[stem] = img

        for ann in self._data['annotations']:
            self._annotations_by_image.setdefault(ann['image_id'], [])
            self._annotations_by_image[ann['image_id']].append(ann)

    def get_available_fragments(self) -> List[str]:
        self._ensure_loaded()
        return sorted(self._image_by_stem.keys())

    def load_image(self, fragment_id: str) -> Optional[cv2.Mat]:
        self._ensure_loaded()
        img_info = self._image_by_stem.get(fragment_id)
        if img_info is None:
            return None
        path = self.coco_dir / self.IMAGES_DIR / img_info['file_name']
        return cv2.imread(str(path))

    def load_annotation(self, fragment_id: str, tablet: Tablet) -> Optional[Boxes]:
        self._ensure_loaded()
        img_info = self._image_by_stem.get(fragment_id)
        if img_info is None:
            return None

        boxes = Boxes(tablet=tablet)
        for ann in self._annotations_by_image.get(img_info['id'], []):
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            sign_name = self._category_names[ann['category_id']]
            sign = SignResolver.resolve(sign_name, expected_type='SIGN')
            boxes.append(Box(
                x1=float(x),
                y1=float(y),
                x2=float(x + w),
                y2=float(y + h),
                score=1.0,
                sign=sign,
                tablet=tablet,
            ))
        return boxes if boxes else None


class EBLAPISource:
    BASE_URL = "https://ebl.badw.de/api"

    def __init__(self, timeout: int = 60, retries: int = 3):
        self.timeout = timeout
        self.retries = retries
        self._fragment_cache = {}

    def get_fragment_data(self, fragment_id: str) -> Optional[dict]:
        if fragment_id in self._fragment_cache:
            return self._fragment_cache[fragment_id]

        url = f"{self.BASE_URL}/fragments/{fragment_id}"
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    self._fragment_cache[fragment_id] = data
                    return data
                last_err = f"HTTP {response.status_code}"
            except requests.RequestException as e:
                last_err = str(e)
        print(
            f"API request failed for fragment {fragment_id} "
            f"after {self.retries + 1} attempts: {last_err}"
        )
        return None


class SubtabletEBLAPISource(EBLAPISource):
    def get_fragment_data(self, fragment_id: str) -> Optional[dict]:
        api_id = re.sub(r'-\d+$', '', fragment_id)
        return super().get_fragment_data(api_id)


class SignTextParser:
    CONTAINER_TYPES = {
        'Word',
        'AkkadianWord',
        'GreekWord',
        'LoneDeterminative',
        'Determinative',
    }

    @staticmethod
    def _is_broken_away(token: dict) -> bool:
        return 'BROKEN_AWAY' in token.get('enclosureType', [])

    @staticmethod
    def _extract_signs_from_token(
        token: dict,
        filter_broken: bool = True,
    ) -> List[Tuple[str, int]]:
        signs = []
        token_type = token.get('type', '')

        # Containers can mix broken and preserved child signs.
        if token_type in SignTextParser.CONTAINER_TYPES:
            for part in token.get('parts', []):
                signs.extend(SignTextParser._extract_signs_from_token(part, filter_broken))
            return signs

        if filter_broken and SignTextParser._is_broken_away(token):
            return signs

        if token_type in ['Reading', 'Logogram', 'Number']:
            name = token.get('name', '')
            sub_index = token.get('subIndex', 1)
            if sub_index is None:
                sub_index = 1
            if name:
                signs.append((name, sub_index))
        elif token_type == 'CompoundGrapheme':
            if filter_broken and 'BROKEN_AWAY' in token.get('enclosureType', []):
                return signs
            clean = token.get('cleanValue', '')
            if clean:
                signs.append((clean, 0))
        elif token_type == 'Grapheme':
            name = token.get('name', '')
            if name:
                signs.append((name, 0))
        elif token_type == 'UnclearSign':
            signs.append(('X', 0))
        elif token_type in ['Variant', 'Variant2']:
            for variant_token in token.get('tokens', []):
                signs.extend(
                    SignTextParser._extract_signs_from_token(variant_token, filter_broken)
                )
                break
        elif token_type == 'Divider':
            divider = token.get('divider', '')
            if divider:
                signs.append((divider, 0))
        elif 'parts' in token:
            for part in token['parts']:
                signs.extend(SignTextParser._extract_signs_from_token(part, filter_broken))

        return signs

    @staticmethod
    def _resolve_sign_token(
        name: str,
        sub_index: int,
        sign_resolver: 'SignAPIResolver' = None,
    ) -> Optional[str]:
        if name == 'X':
            return 'UnclearSign'

        if sub_index == 0:
            return name

        if sign_resolver is not None:
            resolved = sign_resolver.resolve(name, sub_index)
            if resolved:
                return resolved

        try:
            sign = SignResolver.resolve(name.upper(), expected_type='SIGN')
            if sign.name != 'UnclearSign':
                return sign.name
        except Exception:
            pass

        return name.upper()

    @staticmethod
    def parse_text_lines(
        text_data: dict,
        filter_broken: bool = True,
        sign_resolver: 'SignAPIResolver' = None,
    ) -> List[List[str]]:
        if not text_data or 'lines' not in text_data:
            return []

        result_lines = []

        for line in text_data['lines']:
            if line.get('type', '') != 'TextLine':
                continue
            if 'content' not in line:
                continue

            line_signs = []
            for token in line['content']:
                sign_tuples = SignTextParser._extract_signs_from_token(
                    token, filter_broken
                )
                for name, sub_index in sign_tuples:
                    resolved = SignTextParser._resolve_sign_token(
                        name, sub_index, sign_resolver
                    )
                    if resolved:
                        line_signs.append(resolved)

            if line_signs:
                result_lines.append(line_signs)

        return result_lines
