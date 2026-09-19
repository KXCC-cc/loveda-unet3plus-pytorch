"""LoveDA 数据读取与训练、推理共用的图像预处理。"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.constants import (
    map_raw_mask,
    validate_raw_mask,
)
from utils.transforms import (
    preprocess_image,
    transform_image_and_mask,
    validate_image_size,
)


class LoveDADataset(Dataset):
    """读取 root/{Train,Val,Test}/{Rural,Urban} 下的 LoveDA PNG。

    Train / Val 返回 (image, mask)：
        image: float32，[3, H, W]，完成 ImageNet normalization。
        mask: long，[H, W]，原始 0 → 255，原始 1~7 → 0~6。

    Train 默认原分辨率随机 crop，并同步做翻转/直角旋转。
    Val / Test 默认保持完整图像，禁止随机 crop 和训练增强；全图 loader 使用 batch=1。
    spatial_mode='resize' 保留整图缩放的对照策略，mask 始终使用 nearest。
    Test 没有真实标签，返回 (image, str(image_path))。
    samples 保留每个原始文件一项，samples_per_image 只扩展训练迭代长度。
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "Train",
        image_size: Tuple[int, int] = (256, 256),
        spatial_mode: str = "crop",
        augment: Optional[bool] = None,
        samples_per_image: int = 1,
        color_jitter: float = 0.0,
    ) -> None:
        self.root = Path(root)
        if split not in ("Train", "Val", "Test"):
            raise ValueError("split 必须是 'Train'、'Val' 或 'Test'。")
        self.split = split
        self.image_size = validate_image_size(image_size)
        if spatial_mode not in ("crop", "resize", "full"):
            raise ValueError("spatial_mode 必须是 'crop'、'resize' 或 'full'。")
        # 默认参数 crop 只适用于 Train；Val/Test 自动改为确定性的 full。
        self.spatial_mode = (
            "full" if split != "Train" and spatial_mode == "crop"
            else spatial_mode
        )
        self.augment = split == "Train" if augment is None else bool(augment)
        if split != "Train" and self.augment:
            raise ValueError("Val/Test 不允许训练增强。")
        if isinstance(samples_per_image, bool) or not isinstance(
            samples_per_image, int
        ):
            raise TypeError("samples_per_image 必须是正整数。")
        if samples_per_image <= 0:
            raise ValueError("samples_per_image 必须大于 0。")
        if split != "Train" and samples_per_image != 1:
            raise ValueError("Val/Test 的 samples_per_image 必须为 1，确保每张图只计一次。")
        if not 0.0 <= color_jitter <= 1.0:
            raise ValueError("color_jitter 必须位于 [0, 1]，0 表示关闭。")
        if split != "Train" and color_jitter != 0.0:
            raise ValueError("Val/Test 不允许颜色增强。")
        self.samples_per_image = samples_per_image
        self.color_jitter = float(color_jitter)
        self.has_masks = split != "Test"
        self.samples: list[Tuple[Path, Optional[Path]]] = []

        for region in ("Rural", "Urban"):
            region_dir = self.root / split / region
            image_dir = region_dir / "images_png"
            if not image_dir.is_dir():
                raise FileNotFoundError(f"图像目录不存在：{image_dir}")
            image_paths = sorted(
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".png"
            )
            if not image_paths:
                raise ValueError(f"图像目录中没有 PNG 文件：{image_dir}")

            mask_paths = {}
            if self.has_masks:
                mask_dir = region_dir / "masks_png"
                if not mask_dir.is_dir():
                    raise FileNotFoundError(f"标签目录不存在：{mask_dir}")
                mask_paths = {
                    path.name: path
                    for path in mask_dir.iterdir()
                    if path.is_file() and path.suffix.lower() == ".png"
                }
                image_names = {path.name for path in image_paths}
                mask_names = set(mask_paths)
                missing_masks = sorted(image_names - mask_names)
                extra_masks = sorted(mask_names - image_names)
                if missing_masks or extra_masks:
                    raise ValueError(
                        f"{region_dir} 的 image/mask 文件名不匹配。"
                        f"缺少 mask 的图像（最多列出 5 个）：{missing_masks[:5]}；"
                        f"没有对应图像的 mask（最多列出 5 个）：{extra_masks[:5]}"
                    )

            for image_path in image_paths:
                mask_path: Optional[Path] = (
                    mask_paths[image_path.name] if self.has_masks else None
                )
                self.samples.append((image_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples) * self.samples_per_image

    def __getitem__(
        self, index: int,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, str]]:
        """读取原始文件，同步变换后返回图像与标签或 Test 文件路径。"""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("LoveDADataset 索引越界。")
        # 同一原图每轮可抽取多个随机 crop，不复制文件、不预先生成裁剪数据。
        image_path, mask_path = self.samples[index % len(self.samples)]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        raw_mask = None
        if self.has_masks:
            with Image.open(mask_path) as mask_image:
                if mask_image.size != image.size:
                    raise ValueError(
                        f"原始图像与标签尺寸不同：{image_path}={image.size}，"
                        f"{mask_path}={mask_image.size}"
                    )
                # 不 convert('L')：P 模式的调色板索引本身就是类别值。
                raw_mask = np.array(mask_image)
            try:
                validate_raw_mask(raw_mask)
            except (TypeError, ValueError) as error:
                raise ValueError(f"标签文件无效：{mask_path}；{error}") from error

        image, raw_mask = transform_image_and_mask(
            image,
            raw_mask,
            image_size=self.image_size,
            spatial_mode=self.spatial_mode,
            augment=self.augment,
            color_jitter=self.color_jitter,
        )
        image_tensor = preprocess_image(image)  # 只归一化，不再次缩放。
        if not self.has_masks:
            return image_tensor, str(image_path)
        mask_tensor = torch.from_numpy(map_raw_mask(raw_mask))  # long，[H, W]。
        return image_tensor, mask_tensor
