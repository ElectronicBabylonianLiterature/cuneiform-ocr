import base64
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


def _decode_base64_gray(image: str) -> np.ndarray:
    if not image:
        raise ValueError("empty image payload")
    if image.startswith("data:image/"):
        image = image.split(",", 1)[1]
    raw = base64.b64decode(image)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("L"))

# next generate source class
class DataSource:
    def key(self) -> str:
        parts = [type(self).__name__]
        form = getattr(self, "form", None)
        if form:
            parts.append(str(form))
        return ":".join(parts)

    def get(self, sign_name: str, period: str) -> Optional[np.ndarray]:
        raise NotImplementedError
    
class PrototypeSource(DataSource):
    SIGNS_API_URL = "https://www.ebl.lmu.de/api/signs"
    IMAGE_SIZE = 512
    PADDING = 20
    PERIOD_FONTS = {
        "Old-Babylonian-Monumental": "SantakkuM.woff",
        "Old-Babylonian-Literature": "OBFreie.woff",
        "Old-Babylonian-Cursive": "Santakku.woff",
        "Hittite": "UllikummiA.woff",
        "Neo-Assyrian": "Assurbanipal.woff",
        "Neo-Babylonian": "Esagil.woff",
    }
    
    def __init__(self):
        pass
    
    def get(self, sign_name: str, period: str):
        FONT_DIR = Path(__file__).parent / "fonts"
        if period not in self.PERIOD_FONTS:
            src_period = "Neo-Assyrian"
            print(f"Warning: period '{period}' not found, using '{src_period}'")
        else:
            src_period = period
        font_file = FONT_DIR / self.PERIOD_FONTS[src_period]

        from PIL import Image, ImageDraw, ImageFont

        response = requests.get(
            f"{self.SIGNS_API_URL}/{requests.utils.quote(sign_name, safe='')}",
            timeout=1,
        )
        response.raise_for_status()
        unicode_values = response.json().get("unicode", [])

        if not unicode_values:
            print(f"Warning: sign '{sign_name}' has no Unicode value")
            return None

        text = "".join(chr(codepoint) for codepoint in unicode_values)

        font = ImageFont.truetype(str(font_file), self.IMAGE_SIZE)
        probe = Image.new("RGB", (1, 1), "white")
        draw = ImageDraw.Draw(probe)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = max(self.IMAGE_SIZE, bbox[2] - bbox[0] + 2 * self.PADDING)
        height = max(self.IMAGE_SIZE, bbox[3] - bbox[1] + 2 * self.PADDING)

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        x = (width - (bbox[2] - bbox[0])) // 2 - bbox[0]
        y = (height - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((x, y), text, font=font, fill="black")

        gray = img.convert("L")
        bbox = gray.point(lambda p: 255 - p).getbbox()
        cropped = gray.crop(bbox) if bbox else gray
        padded = Image.new(
            "L",
            (cropped.width + 2 * self.PADDING, cropped.height + 2 * self.PADDING),
            255,
        )
        padded.paste(cropped, (self.PADDING, self.PADDING))
        resample = getattr(Image, "Resampling", Image).LANCZOS
        resized = padded.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), resample)

        return np.array(resized)


class EBLMongoCanonicalSource(DataSource):
    def __init__(
        self,
        mongodb_uri: Optional[str],
        db_name: str = "ebl",
        form: str = "canonical1",
        require_centroid: bool = True,
    ):
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.form = form
        self.require_centroid = require_centroid

    def get(self, sign_name: str, period: str) -> Optional[np.ndarray]:
        from pymongo import MongoClient

        annotation_match: Dict[str, Any] = {
            "data.type": "HasSign",
            "data.signName": sign_name,
            "pcaClustering.form": self.form,
            "croppedSign.imageId": {"$exists": True, "$ne": None},
        }
        if self.require_centroid:
            annotation_match["pcaClustering.isCentroid"] = True
        pipeline: List[Dict[str, Any]] = [
            {"$match": {"annotations": {"$elemMatch": annotation_match}}},
            {
                "$lookup": {
                    "from": "fragments",
                    "localField": "fragmentNumber",
                    "foreignField": "_id",
                    "as": "fragment",
                }
            },
            {"$unwind": "$fragment"},
            {"$match": {"fragment.script.period": period}},
            {"$unwind": "$annotations"},
            {"$match": {
                f"annotations.{key}": value
                for key, value in annotation_match.items()
            }},
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
                "$sort": {
                    "annotations.pcaClustering.isMain": -1,
                    "annotations.pcaClustering.clusterSize": -1,
                    "annotations.pcaClustering.clusterRank": 1,
                }
            },
            {"$limit": 1},
            {
                "$project": {
                    "_id": 0,
                    "fragmentNumber": 1,
                    "signName": "$annotations.data.signName",
                    "image": "$imageDoc.image",
                    "period": "$fragment.script.period",
                    "annotationId": "$annotations.data.id",
                    "pcaClustering": "$annotations.pcaClustering",
                }
            },
        ]
        client = MongoClient(self.mongodb_uri)
        try:
            item = next(
                client[self.db_name]["annotations"].aggregate(
                    pipeline, allowDiskUse=True
                ),
                None,
            )
            return _decode_base64_gray(item["image"]) if item else None
        finally:
            client.close()


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

    def __init__(
        self,
        timeout: int = 60,
        retries: int = 3,
        strip_subtablet_suffix: bool = False,
    ):
        self.timeout = timeout
        self.retries = retries
        self.strip_subtablet_suffix = strip_subtablet_suffix
        self._fragment_cache = {}

    def get_fragment_data(self, fragment_id: str) -> Optional[dict]:
        api_fragment_id = self._api_fragment_id(fragment_id)
        if api_fragment_id in self._fragment_cache:
            return self._fragment_cache[api_fragment_id]

        url = f"{self.BASE_URL}/fragments/{api_fragment_id}"
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    self._fragment_cache[api_fragment_id] = data
                    return data
                last_err = f"HTTP {response.status_code}"
            except requests.RequestException as e:
                last_err = str(e)
        print(
            f"API request failed for fragment {api_fragment_id} "
            f"after {self.retries + 1} attempts: {last_err}"
        )
        return None

    def _api_fragment_id(self, fragment_id: str) -> str:
        if self.strip_subtablet_suffix:
            return re.sub(r'-\d+$', '', fragment_id)
        return fragment_id


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
