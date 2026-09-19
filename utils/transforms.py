"""同步几何变换：图像和原始 mask 始终使用同一个裁剪框与翻转方向。"""

import random
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance

from utils.constants import IMAGENET_MEAN, IMAGENET_STD, RAW_IGNORE_INDEX


def validate_image_size(image_size: Tuple[int, int]) -> Tuple[int, int]:
    """尺寸统一使用 (高度, 宽度)，Pillow resize 时再交换顺序。"""
    if len(image_size) != 2:
        raise ValueError("image_size 必须是 (height, width)。")
    height, width = image_size
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in image_size
    ):
        raise TypeError("image_size 中的高度、宽度必须是整数。")
    if height <= 0 or width <= 0:
        raise ValueError("image_size 中的高度、宽度必须大于 0。")
    return height, width


def preprocess_image(
    image: Image.Image,
    image_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """RGB → 可选 bilinear resize → [0, 1] → ImageNet normalization。

    返回 float32 [3, H, W]。image_size=None 保持原尺寸，适合全图滑窗；
    resize 对照策略显式传入训练尺寸，训练与推理使用同一预处理。
    """
    image = image.convert("RGB")
    if image_size is not None:
        height, width = validate_image_size(image_size)
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    array = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def transform_image_and_mask(
    image: Image.Image,
    raw_mask: Optional[np.ndarray],
    image_size: Tuple[int, int],
    spatial_mode: str,
    augment: bool = False,
    color_jitter: float = 0.0,
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """先做 crop/resize/full，再做可选同步增强；标签仍保留原始 0~7。

    原始 mask 的尺寸、类型和标签范围由 Dataset 在进入这里前检查。
    只为全 ignore 的 crop 有限重采样，不对类别比例做强制筛选。
    """
    crop_height, crop_width = validate_image_size(image_size)
    image = image.convert("RGB")

    if spatial_mode == "crop":
        width, height = image.size
        pad_height = max(crop_height - height, 0)
        pad_width = max(crop_width - width, 0)
        if pad_height or pad_width:
            # 小图只在右侧/底部补边。图像复制边缘，mask 补原始 ignore 值 0。
            padded_image = np.pad(
                np.array(image),
                ((0, pad_height), (0, pad_width), (0, 0)),
                mode="edge",
            )
            image = Image.fromarray(padded_image)
            if raw_mask is not None:
                raw_mask = np.pad(
                    raw_mask,
                    ((0, pad_height), (0, pad_width)),
                    mode="constant",
                    constant_values=RAW_IGNORE_INDEX,
                )
        width, height = image.size
        cropped_mask = None
        for _ in range(10):
            top = random.randint(0, height - crop_height)
            left = random.randint(0, width - crop_width)
            if raw_mask is None:
                break
            cropped_mask = raw_mask[
                top : top + crop_height, left : left + crop_width
            ]
            if np.any(cropped_mask != RAW_IGNORE_INDEX):
                break
        # 即使整张原图都是 ignore，也只尝试 10 次，随后由损失/训练循环处理。
        image = image.crop((left, top, left + crop_width, top + crop_height))
        raw_mask = cropped_mask
    elif spatial_mode == "resize":
        image = image.resize(
            (crop_width, crop_height), resample=Image.Resampling.BILINEAR
        )
        if raw_mask is not None:
            mask_image = Image.fromarray(raw_mask.astype(np.uint8))
            raw_mask = np.array(
                mask_image.resize(
                    (crop_width, crop_height), resample=Image.Resampling.NEAREST
                )
            )
    elif spatial_mode != "full":
        raise ValueError("spatial_mode 必须是 'crop'、'resize' 或 'full'。")

    if augment:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if raw_mask is not None:
                raw_mask = np.fliplr(raw_mask)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if raw_mask is not None:
                raw_mask = np.flipud(raw_mask)

        # 矩形 crop 只允许 0/180 度，避免增强后 H/W 对调而不能组成 batch。
        turns = random.choice(
            (0, 1, 2, 3) if image.width == image.height else (0, 2)
        )
        rotations = {
            1: Image.Transpose.ROTATE_90,
            2: Image.Transpose.ROTATE_180,
            3: Image.Transpose.ROTATE_270,
        }
        if turns:
            image = image.transpose(rotations[turns])
            if raw_mask is not None:
                raw_mask = np.rot90(raw_mask, k=turns)

        # 颜色增强只作用于 RGB；默认 0 关闭，避免改变遥感场景颜色过多。
        if color_jitter > 0:
            for enhancer in (
                ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color,
            ):
                factor = random.uniform(1.0 - color_jitter, 1.0 + color_jitter)
                image = enhancer(image).enhance(factor)

    return image, raw_mask
