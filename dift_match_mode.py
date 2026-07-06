

from dataclasses import dataclass
import importlib
import importlib.machinery
import importlib.util
import os
import sys
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import types
from argparse import Namespace

import cv2
import numpy as np
import torch
from PIL import Image

from data_processing.sign_resolver import Sign, SignResolver
from sign_alignment.data_source import DataSource, PrototypeSource


import sign_alignment.dift_src.dift as dift


@dataclass
class ImageView:
    src_img: Any
    pil_rgb: Image.Image
    dtype: Optional[Any] = None
    shape: Optional[Tuple[int, ...]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    @classmethod
    def from_any(cls, img: Any, *, assume_bgr: bool = False) -> "ImageView":
        if isinstance(img, ImageView):
            return img
        if isinstance(img, Image.Image):
            arr = np.asarray(img)
            pil_rgb = img.convert("RGB")
            return cls(
                src_img=img,
                pil_rgb=pil_rgb,
                dtype=arr.dtype,
                shape=arr.shape,
                min_value=float(arr.min()) if arr.size else None,
                max_value=float(arr.max()) if arr.size else None,
            )

        elif isinstance(img, torch.Tensor):
            tensor = img.detach().cpu()
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)
            arr = tensor.numpy()
        else:
            arr = np.asarray(img)

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            rgb = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if assume_bgr else arr
        elif arr.ndim == 3 and arr.shape[2] == 4:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB if assume_bgr else cv2.COLOR_RGBA2RGB)
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
        
        pil_rgb=Image.fromarray(rgb).convert("RGB")
        return cls(
            src_img=img,
            pil_rgb=pil_rgb,
            dtype=arr.dtype,
            shape=arr.shape,
            min_value=float(arr.min()) if arr.size else None,
            max_value=float(arr.max()) if arr.size else None,
        )

    def as_pil(self) -> Image.Image:
        return self.pil_rgb
    
    def as_rgb_numpy(self) -> np.ndarray:
        return np.asarray(self.pil_rgb)

    def as_bgr_numpy(self) -> np.ndarray:
        return cv2.cvtColor(self.as_rgb_numpy(), cv2.COLOR_RGB2BGR)

    def as_gray_numpy(self) -> np.ndarray:
        return cv2.cvtColor(self.as_rgb_numpy(), cv2.COLOR_RGB2GRAY)

    def as_tensor_chw(self) -> torch.Tensor:
        return torch.from_numpy(self.as_rgb_numpy()).permute(2, 0, 1)

    def save(self, path: str | Path) -> None:
        self.pil_rgb.save(Path(path))

@dataclass
class DiftMatchResult:
    similarity_tensor: torch.Tensor # H, W, H, W


@dataclass
class DiftContext:
    source: DataSource
    feature_dir: str

    feature_cache: Dict[str, torch.Tensor]

    featurizer: dift.SDFeaturizer
    dift_wrapper: dift.DiftWrapper = None

    img_size: int = 512

    def __post_init__(self):
        self.dift_wrapper = dift.DiftWrapper(Namespace(prompt="", img_size=self.img_size), dift=self.featurizer)

    def _make_disk_path(self, sid: str):
        import hashlib
        import re

        feature_dir = Path(self.feature_dir).expanduser()
        assert feature_dir.is_dir(), f"Feature directory does not exist: {feature_dir}"
        
        readable = re.sub(r"[^\w.@+-]+", "_", sid, flags=re.UNICODE).strip("_") # make sid as readable filename
        readable = readable[:160] or "value"
        digest = hashlib.sha1(sid.encode("utf-8")).hexdigest()[:10] # add digest to make name unique
        path = feature_dir / f"{readable}_{digest}.pt"
        return path

    def load_from_disk(self, sid: str) -> torch.Tensor:
        
        path = self._make_disk_path(sid)
        if path is None or not path.exists():
            return None

        feature = torch.load(path, map_location="cpu", weights_only=False)
        return feature

    def save_to_disk(self, sid: str, feature: torch.Tensor):
        path = self._make_disk_path(sid)
        torch.save(feature, path)


    def get_sign_feature(self, sign: Sign, period: str) -> torch.Tensor: # C, H, W
        sid = f"{self.source.key()}/{period}/{sign.name}"

        feature_map = self.feature_cache.get(sid)
        if feature_map is not None:
            return feature_map
        
        feature_map = self.load_from_disk(sid)
        if feature_map is not None:
            self.feature_cache[sid] = feature_map
            return feature_map

        img = self.source.get(sign.name, period)
        image = ImageView.from_any(img)

        # -- debug --
        debug_img_path = Path("/tmp/dift_match_mode_debug.png")
        image.save(debug_img_path)
        print(f"Saved debug image to {debug_img_path}")
        print(
            "Image stats: "
            f"shape={image.shape}, dtype={image.dtype}, "
            f"min={image.min_value}, max={image.max_value}"
        )
        # raise SystemExit(0)
        # -- -- \

        feature_map = self.featurize_image(image) 
        self.feature_cache[sid] = feature_map
        self.save_to_disk(sid, feature_map)

        return feature_map


    def featurize_image(self, image) -> torch.Tensor:
        image_view = ImageView.from_any(image)
        with torch.no_grad():
            feature_map = self.dift_wrapper.featurize(image_view.as_pil()).detach()
            return feature_map
    
    def match(self, src_feature: torch.Tensor, target_feature: torch.Tensor) -> DiftMatchResult:
        similarity_tensor = self.dift_wrapper.get_similarity_tensor(src_feature, target_feature)
        return DiftMatchResult(
            similarity_tensor=similarity_tensor
            )

        
    

DIFT_CHECKPOINT = os.path.expanduser("~/erc-src/ProtoSnap/weights/SD_with_prompt")

CANONICAL_FEATURE_DIR = "~/erc-work-data/signs_alignment_data/precompute_feautures_2/"



context = DiftContext(
    source=PrototypeSource(),
    feature_dir=CANONICAL_FEATURE_DIR,
    feature_cache={},
    featurizer=dift.SDFeaturizer(sd_id=DIFT_CHECKPOINT),
)


period = "Middle_Assyrian"


# ========= check feature map calculation ==========
sign = SignResolver.from_name("AN")
feature_an = context.get_sign_feature(sign, period)

OLD_FEATURE_DIR = Path("~/erc-work-data/signs_alignment_data/precompute_feautures").expanduser()
# Old format: {sign_num}_{sign_name}_{sign_hash}__{period}_{period_hash}__{source}_{source_hash}.pt
feature_an_previous_path = OLD_FEATURE_DIR / "AN_bce92920ac__Middle_Assyrian_e62fdb670d__PrototypeSource_d938585c48.pt"
feature_an_previous = torch.load(feature_an_previous_path, map_location="cpu", weights_only=False)["feature_map"]
print(f"Loaded previous feature: shape={feature_an_previous.shape}, dtype={feature_an_previous.dtype}")

feature_3an_previous_path = OLD_FEATURE_DIR / "3_AN_691aa734a3__Middle_Assyrian_e62fdb670d__canonical1_81557f9709.pt"
feature_3an_previous = torch.load(feature_3an_previous_path, map_location="cpu", weights_only=False)["feature_map"]
print(f"Loaded previous feature for 3_AN: shape={feature_3an_previous.shape}, dtype={feature_3an_previous.dtype}")

print(feature_an[0, :, :])

print(feature_an_previous[0, :, :])

print(feature_3an_previous[0, :, :])

# calculate the cosine similarity between the two features
cosine_similarity_an_prean = torch.nn.functional.cosine_similarity(feature_an.flatten(), feature_an_previous.flatten(), dim=0)
print(f"Cosine similarity between new and previous feature: {cosine_similarity_an_prean.item()}")

cosine_similarity_an_3an = torch.nn.functional.cosine_similarity(feature_an.flatten(), feature_3an_previous.flatten(), dim=0)
print(f"Cosine similarity between new feature and 3_AN previous feature: {cosine_similarity_an_3an.item()}")

print(f"Feature for sign {sign.name} in period {period}: shape={feature_an.shape}, dtype={feature_an.dtype}")

# print(help(dift.SDFeaturizer))


# =============


note = \
'''
- Runtime 设计
   - 保持 dift_match_mode.py 中简洁 DiftContext 的设计方向，正式实现命名为 DiftRuntime。
   - DiftRuntime 用于取代 sign_alignment/dift_align.py 中现有的 CanonicalFeatureRecord / CanonicalFeatureCache / CanonicalFeatureSet 设计。
   - DiftRuntime 中不保存 period；period 每次通过 get_sign_image / get_sign_feature 等调用传入。
   - 可以接受对 pipeline、pipeline_2 和可视化调用做较大改动，不需要维持旧 CanonicalFeature* 接口。
-  DataSource 与命名
   - 继续支持 EBLMongoCanonicalSource、PrototypeSource 等现有 DataSource。
   - 新 runtime 和 match API 中不再强制使用 canonical 命名，可以统一称为 sign/source/prototype image 与 sign/source feature。
   - 这里的 source image 可以来自 EBLMongoCanonicalSource，也可以来自 PrototypeSource；DIFT 匹配逻辑不应绑定到 canonical 这个概念。
3. Feature cache
   - 放弃旧 feature cache 格式，不做兼容迁移；使用新的 feature cache 命名和 payload 格式，已有特征允许重新计算。
   - 砍掉 CanonicalFeatureRecord / CanonicalFeatureCache / CanonicalFeatureSet 后，直接查询和传递 feature tensor。
   - 特征同时缓存到 runtime.feature_cache 和 disk cache。
   - 去掉 _safe_cache_part 这类一次性辅助函数；使用 dift_match_mode.py 中更直接的 readable + sha1 disk path 思路即可。
   - disk cache 目录应由 runtime 初始化或保存时确保存在，不依赖调用者手工预创建。
- Feature API
   - 用 get_sign_feature(sign, period, source=None) 计算/读取 sign/source/prototype 特征。
   - 用 get_sign_image(sign, period, source=None) 获取对应 sign/source/prototype 原图，供 foreground mask、feature viz、overlay 和 manual figure 使用。
   - 用 featurize_image(image, assume_bgr=False) 计算任意图片或 crop 的特征。
   - crop 应在调用侧通过 Box.crop_image 等方式完成；runtime 只负责 featurize 和 match。
- ImageView
   - 用 ImageView 统一 PIL / numpy / torch.Tensor 输入和 RGB/BGR 转换。
   - ImageView 替代 dift_align.py 里的 _to_rgb、部分 _to_bgr 预处理职责。
   - 调用 featurize_image 时不需要assume_bgr=True之类的参数，所有的opencv处理图像都保持默认的bgr顺序，可以直接asume bgr，只有imageviewer可以操作特殊的rgb转换。
   - 纯可视化层如果仍需要 BGR 输出，可以保留很薄的显示/绘图转换函数，但不要让特征计算再依赖散落的 _to_rgb。
- Match API 与层级
   - 砍掉 _compute_manual_dift_crop_match；调用侧先分别得到 source feature 和 crop feature，再调用 DiftRuntime.match。
   - 砍掉 cache.match -> match_features -> match_dift_features -> _crop_match_result 这类不必要层级。
   - DiftRuntime.match 直接接收 src_feature、dst_feature、src_img_shape、dst_img_shape、config、src_foreground_mask 等必要输入，并返回 DiftMatchResult。
   - DiftRuntime.match 中集中计算 mutual best-buddies、global similarity、sim_withoutbg、RANSAC affine 和最终 score。
- DiftMatchResult
   - 使用新定义的 DiftMatchResult 作为正式返回结果，但需要补全字段。
   - 暂时保留现有 DIFT 匹配指标和计算方式，包括 affine、src_points、dst_points、similarities、inlier_mask、n_matches、n_inliers、mean_inlier_similarity、semantic_score、global_similarity_score、sim_withoutbg、geometry_score、support_score、score、coarse_score、message。
   - DiftMatchResult 必须足够支撑现有可视化、manual match 和 feature coarse alignment。
   - 暂时不改变 score / coarse_score 的计算方式。
- sign_name 检查
   - 原 note 中“去除 sign_name 的 strip 和各种检查”修正为：深层 helper 不再负责 sign_name 清洗和 UI 输入校验。
   - 假定sign_name不会出错，不需要校验处理。
- dift_match_mode.py 的定位
   - dift_match_mode.py 继续作为参考实验脚本保持不变；除本 note2 外，不在这里做正式重构。
   - 正式实现应进入 sign_alignment/dift_align.py 或后续拆分出的包内模块。
- 实施要求
   - 补全必要步骤并修正 note 原要求中不合理或错误的做法。
   - 重构不要求功能完全一致，但必须保持核心功能：source image 获取、feature cache、任意 crop featurize、DIFT match、核心指标、manual match、feature coarse alignment 所需分数。
   - 删除旧类和旧层级前，要同步更新 pipeline.py、pipeline_2.py、__init__.py 和 notebook 入口中暴露/使用的 API。
   - 重构目标为简化、清晰、直接，禁止过度设计。
'''
