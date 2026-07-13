

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
    def from_any(cls, img: Any, *, assume_bgr: bool = True) -> "ImageView":
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
    affine: Optional[np.ndarray]
    src_points: np.ndarray
    dst_points: np.ndarray
    similarities: np.ndarray
    inlier_mask: np.ndarray
    n_matches: int
    n_inliers: int
    mean_inlier_similarity: float
    semantic_score: float
    global_similarity_score: float
    sim_withoutbg: float
    geometry_score: float
    support_score: float
    score: float
    coarse_score: float
    message: str = ""


@dataclass(frozen=True)
class DiftMatchConfig:
    max_matches: int = 300
    min_matches: int = 3
    min_support: int = 12
    ransac_threshold: Optional[float] = None


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
    
    def match(
        self,
        src_feature: torch.Tensor,
        dst_feature: torch.Tensor,
        src_img_shape: Tuple[int, int],
        dst_img_shape: Tuple[int, int],
        config: Optional[DiftMatchConfig] = None,
        src_foreground_mask: Optional[np.ndarray] = None,
    ) -> DiftMatchResult:
        """Match local features and score semantic plus geometric agreement."""
        config = config or DiftMatchConfig()
        src = src_feature.detach().float()
        dst = dst_feature.detach().float()
        device = dst.device
        src = src.to(device=device)

        c = src.shape[0]
        src_flat = src.reshape(c, -1)
        dst_flat = dst.reshape(c, -1)
        sim = src_flat.T @ dst_flat
        global_similarity = 0.5 * (
            sim.max(dim=1).values.mean() + sim.max(dim=0).values.mean()
        )
        sim_withoutbg = _foreground_only_global_similarity(
            sim,
            src_foreground_mask,
            fallback=global_similarity,
        )

        src_to_dst = sim.argmax(dim=1)
        dst_to_src = sim.argmax(dim=0)
        src_idx = torch.arange(sim.shape[0], device=device)
        is_mutual = dst_to_src[src_to_dst] == src_idx
        mutual_src = src_idx[is_mutual]
        mutual_dst = src_to_dst[is_mutual]
        mutual_sim = sim[mutual_src, mutual_dst]

        if mutual_src.numel() > config.max_matches:
            vals, keep = torch.topk(mutual_sim, k=config.max_matches)
            mutual_src = mutual_src[keep]
            mutual_dst = mutual_dst[keep]
            mutual_sim = vals

        src_points = _feature_indices_to_image_points(
            mutual_src.detach().cpu().numpy(),
            src.shape[1],
            src.shape[2],
            src_img_shape[0],
            src_img_shape[1],
        )
        dst_points = _feature_indices_to_image_points(
            mutual_dst.detach().cpu().numpy(),
            dst.shape[1],
            dst.shape[2],
            dst_img_shape[0],
            dst_img_shape[1],
        )
        similarities = mutual_sim.detach().cpu().numpy()
        global_similarity_score = float(global_similarity.detach().cpu())
        sim_withoutbg_score = float(sim_withoutbg.detach().cpu())

        def finish(
            *,
            affine: Optional[np.ndarray] = None,
            inlier_mask: Optional[np.ndarray] = None,
            message: str = "",
        ) -> DiftMatchResult:
            n_matches = len(src_points)
            if inlier_mask is None:
                mask = np.zeros(n_matches, dtype=bool)
            else:
                mask = np.asarray(inlier_mask, dtype=bool)
            n_inliers = int(mask.sum())
            mean_similarity = (
                float(similarities[mask].mean()) if n_inliers else 0.0
            )
            semantic = float(np.clip(mean_similarity, 0.0, 1.0))
            global_score = float(np.clip(global_similarity_score, 0.0, 1.0))
            no_bg_score = float(np.clip(sim_withoutbg_score, 0.0, 1.0))
            geometry = float(n_inliers / n_matches) if n_matches else 0.0
            support = float(min(1.0, n_inliers / max(1, config.min_support)))
            coarse = float(geometry * support)
            score = float(semantic * np.sqrt(coarse))
            return DiftMatchResult(
                affine=affine,
                src_points=src_points,
                dst_points=dst_points,
                similarities=similarities,
                inlier_mask=mask,
                n_matches=n_matches,
                n_inliers=n_inliers,
                mean_inlier_similarity=mean_similarity,
                semantic_score=semantic,
                global_similarity_score=global_score,
                sim_withoutbg=no_bg_score,
                geometry_score=geometry,
                support_score=support,
                score=score,
                coarse_score=coarse,
                message=message,
            )

        n_matches = len(src_points)
        if n_matches < config.min_matches:
            return finish(
                message=f"need at least {config.min_matches} mutual matches"
            )

        threshold = config.ransac_threshold
        if threshold is None:
            threshold = max(3.0, 0.06 * max(dst_img_shape))

        src32 = src_points.astype(np.float32)
        dst32 = dst_points.astype(np.float32)
        affine = inliers = None
        for estimator in (cv2.estimateAffine2D, cv2.estimateAffinePartial2D):
            affine, inliers = estimator(
                src32,
                dst32,
                method=cv2.RANSAC,
                ransacReprojThreshold=float(threshold),
                maxIters=2000,
                confidence=0.99,
                refineIters=10,
            )
            if affine is not None:
                break
        if affine is None:
            return finish(message="affine RANSAC failed")

        inlier_mask = (
            inliers.ravel().astype(bool)
            if inliers is not None
            else np.ones(n_matches, dtype=bool)
        )
        return finish(affine=affine, inlier_mask=inlier_mask)

        
    

def _foreground_only_global_similarity(
    sim: torch.Tensor,
    src_foreground_mask: Optional[np.ndarray],
    fallback: torch.Tensor,
) -> torch.Tensor:
    if src_foreground_mask is None:
        return fallback

    mask = torch.as_tensor(
        np.asarray(src_foreground_mask, dtype=bool).reshape(-1),
        device=sim.device,
        dtype=torch.bool,
    )
    if mask.numel() != sim.shape[0] or not bool(mask.any()):
        return fallback

    foreground_sim = sim[mask]
    return 0.5 * (
        foreground_sim.max(dim=1).values.mean()
        + foreground_sim.max(dim=0).values.mean()
    )


def _feature_indices_to_image_points(
    indices: np.ndarray,
    grid_h: int,
    grid_w: int,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    if indices.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    ys = indices // grid_w
    xs = indices % grid_w
    pts_x = (xs + 0.5) / grid_w * img_w
    pts_y = (ys + 0.5) / grid_h * img_h
    return np.stack([pts_x, pts_y], axis=1).astype(np.float32)


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

# =============

from sign_alignment.data_source import LocalDataSource
from sign_alignment.detector import ModelConfig, TabletImageDetector
from sign_alignment.tablet import Tablet


ANNOTATIONS_DIR = os.path.expanduser("~/erc-work-data/data-of-cuneiform-ocr-data/filtered_annotations")
CONFIG_FILE = "configs/detr.py"
CHECKPOINT_FILE = os.path.expanduser("~/erc-work-data/retrained_models/detr-173/epoch_1000.pth")
SCORE_THRESHOLD = 0.5

test_detector = TabletImageDetector(
    score_threshold=SCORE_THRESHOLD,
    model_config=ModelConfig(
        config_file=CONFIG_FILE,
        checkpoint_file=CHECKPOINT_FILE,
        device="auto",
    ),
    keep_crops=True,
    is_crop_itself=False,
)
test_root_tablet = Tablet(
    img=LocalDataSource(ANNOTATIONS_DIR).load_image("YBC.12860"),
    name="YBC.12860",
)
test_detector.detect(test_root_tablet)
test_tablet = test_detector.get_crop_tablets()[5]

# show test_tablet.img
ImageView.from_any(test_tablet.img).as_pil().save("/tmp/test_tablet.png")
print(f"Saved test_tablet image to /tmp/test_tablet.png")

# Crop detected signs in crop_5.
test_boxes = test_detector.get_crop_boxes()[5]
test_sign_names = ("AN", "TUR", "LU")
test_boxes_by_sign = {name: next(b for b in test_boxes if b.sign_name == name) for name in test_sign_names}
test_crop_imgs_by_sign = {name: box.crop_image() for name, box in test_boxes_by_sign.items()}

for sign_name, crop_img in test_crop_imgs_by_sign.items():
    crop_path = f"/tmp/test_box_crop_{sign_name}.png"
    box = test_boxes_by_sign[sign_name]
    ImageView.from_any(crop_img).as_pil().save(crop_path)
    print(f"Saved {sign_name} detected sign crop to {crop_path}; bbox={box.bbox}, score={box.score:.4f}")

box_an = test_boxes_by_sign["AN"]
box_tur = test_boxes_by_sign["TUR"]
box_lu = test_boxes_by_sign["LU"]
crop_img_an = test_crop_imgs_by_sign["AN"]
crop_img_tur = test_crop_imgs_by_sign["TUR"]
crop_img_lu = test_crop_imgs_by_sign["LU"]

# Set this by hand for a random crop area: x1, y1, x2, y2 in crop_5 coordinates.
manual_crop_xyxy = (500, 500, 920, 820)
x1, y1, x2, y2 = manual_crop_xyxy
crop_img_manual = test_tablet.img[y1:y2, x1:x2].copy()
ImageView.from_any(crop_img_manual).as_pil().save("/tmp/test_box_crop_manual.png")
print(f"Saved manual crop to /tmp/test_box_crop_manual.png; bbox={list(manual_crop_xyxy)}")


prototype_an = context.source.get(sign_name="AN", period=period)



# cal features on prototype and crop
prototype_an_feature = context.featurize_image(prototype_an)
crop_img_an_feature = context.featurize_image(crop_img_an)

match_result_an = context.match(
    src_feature=prototype_an_feature,
    dst_feature=crop_img_an_feature,
    src_img_shape=ImageView.from_any(prototype_an).shape[:2],
    dst_img_shape=crop_img_an.shape[:2],
)



print(prototype_an_feature.shape, crop_img_an_feature.shape)


# saliency map
def save_crop_saliency(prototype_img, prototype_feature, crop_img, crop_feature, name: str):
    proto_gray = ImageView.from_any(prototype_img).as_gray_numpy()
    grid_h, grid_w = prototype_feature.shape[1:]
    proto_mask = cv2.resize(
        proto_gray,
        (grid_w, grid_h),
        interpolation=cv2.INTER_AREA,
    ) != 255

    src = prototype_feature.detach().float()
    dst = crop_feature.detach().float().to(device=src.device)
    mask = torch.as_tensor(proto_mask, device=src.device, dtype=torch.bool)
    sim = torch.einsum("cij,ckl->ijkl", src, dst)
    saliency = sim[mask, :, :].mean(dim=0) - sim[~mask, :, :].mean(dim=0)
    saliency -= saliency.min()
    saliency /= saliency.max().clamp_min(1e-6)

    
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(2, 2))
    saliency_np = saliency.detach().cpu().numpy()
    saliency_eq = clahe.apply((saliency_np * 255).astype(np.uint8)).astype(np.float32)
    saliency_eq -= saliency_eq.mean()
    saliency_eq = np.clip(saliency_eq, 0.0, None)
    saliency_eq /= max(float(saliency_eq.max()), 1e-6)

    saliency_eq = saliency.detach().cpu().numpy().astype(np.float32) # bypass equalization for now

    crop_view = ImageView.from_any(crop_img)
    crop_h, crop_w = crop_view.shape[:2]
    saliency_full = cv2.resize(
        saliency_eq,
        (crop_w, crop_h),
        interpolation=cv2.INTER_CUBIC,
    )
    saliency_heat = cv2.applyColorMap(
        (saliency_full * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    saliency_overlay = cv2.addWeighted(
        crop_view.as_bgr_numpy(),
        0.55,
        saliency_heat,
        0.45,
        0,
    )
    heat_path = f"/tmp/crop_img_{name}_saliency.png"
    overlay_path = f"/tmp/crop_img_{name}_saliency_overlay.png"
    cv2.imwrite(heat_path, saliency_heat)
    cv2.imwrite(overlay_path, saliency_overlay)
    print(f"Saved {name} saliency map to {heat_path}")
    print(f"Saved {name} saliency overlay to {overlay_path}")


prototype_imgs_by_name = {
    "an": prototype_an,
    "tur": context.source.get(sign_name="TUR", period=period),
    "lu": context.source.get(sign_name="LU", period=period),
}
prototype_features_by_name = {
    name: context.featurize_image(img)
    for name, img in prototype_imgs_by_name.items()
}
crop_imgs_for_saliency = {
    "an": crop_img_an,
    "tur": crop_img_tur,
    "lu": crop_img_lu,
    "manual": crop_img_manual,
}
crop_features_for_saliency = {
    "an": crop_img_an_feature,
    "tur": context.featurize_image(crop_img_tur),
    "lu": context.featurize_image(crop_img_lu),
    "manual": context.featurize_image(crop_img_manual),
}

for name in ("an", "tur", "lu"):
    save_crop_saliency(
        prototype_imgs_by_name[name],
        prototype_features_by_name[name],
        crop_imgs_for_saliency[name],
        crop_features_for_saliency[name],
        name,
    )

save_crop_saliency(
    prototype_an,
    prototype_an_feature,
    crop_img_manual,
    crop_features_for_saliency["manual"],
    "manual_an",
)



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
