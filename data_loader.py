"""数据集与 DataLoader 封装：图像 → 力场 (image → force map)

功能要点
- ForceFieldDataset：按文件名对齐图像(.png/.jpg)与力场(.npy)对，提供缺失值填补能力。
- fill_missing_force_values：基于最近参考点 + 指数衰减填补缺失值(-1)。
- get_dataloaders：构建训练/测试 DataLoader。
- preprocess_image：单张图像的推理前处理。
"""

import os
import glob
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.ndimage import zoom

from config import DATA_CONFIG, BASE_CONFIG


def fill_missing_force_values(force_matrix: np.ndarray, decay_rate: float = 0.1) -> np.ndarray:
    """填补力场矩阵中的缺失值（值为 -1）。

    算法：收集所有“有效参考点”（值≠-1且≠0），对每个缺失点采用最近参考点值并按距离做指数衰减：
        fill_val = ref_val * exp(-decay_rate * distance)

    参数:
        force_matrix: (H, W) 浮点矩阵，-1 表示缺失。
        decay_rate: 衰减因子，越大表示距离影响越强。

    返回:
        (H, W) float32，缺失值已填补。
    """
    filled_matrix = force_matrix.copy()
    h, w = filled_matrix.shape

    # 收集“有效参考点”：值既不是 -1 也不是 0（0 视为无信息）
    valid_reference_points = []
    for i in range(h):
        for j in range(w):
            val = filled_matrix[i, j]
            if val != -1 and val != 0:
                valid_reference_points.append((i, j, val))

    if not valid_reference_points:
        # 无参考点时，退化为 0 填充（并提示）
        print("警告：当前力场数据无有效参考点（无'非-1且非0'的已知点），所有缺失值填充为0")
        filled_matrix[filled_matrix == -1] = 0
        return filled_matrix.astype(np.float32)

    # 对缺失点逐个填补（小数据量下可接受；若大规模可改为 KDTree 加速）
    missing_positions = np.argwhere(filled_matrix == -1)
    for (i, j) in missing_positions:
        min_distance = float('inf')
        nearest_reference_val = 0.0
        for (ref_i, ref_j, ref_val) in valid_reference_points:
            distance = math.hypot(i - ref_i, j - ref_j)
            if distance < min_distance:
                min_distance = distance
                nearest_reference_val = ref_val
        # 距离为0时直接赋值，避免 0 除或 exp(0)
        fill_val = nearest_reference_val if min_distance == 0 else nearest_reference_val * np.exp(-decay_rate * min_distance)
        filled_matrix[i, j] = fill_val
    return filled_matrix.astype(np.float32)


class ForceFieldDataset(Dataset):
    """成对数据集：图像（RGB）和力场（2D float map）。"""

    def __init__(
        self,
        image_dir: str,
        force_dir: str,
        image_size=(1080, 1920),
        transform=None,
        fill_missing: bool = True,
        decay_rate: float = 0.1,
    ):
        self.image_dir = image_dir
        self.force_dir = force_dir
        self.image_size = image_size
        self.transform = transform
        self.fill_missing = fill_missing
        self.decay_rate = decay_rate

        self.image_paths = self._get_sorted_paths(image_dir, [".png", ".jpg"])
        self.force_paths = self._get_sorted_paths(force_dir, [".npy"])
        assert len(self.image_paths) == len(self.force_paths), \
            f"图像数量({len(self.image_paths)})与力场标签数量({len(self.force_paths)})不匹配"

        self._precheck_force_data()

    def _get_sorted_paths(self, dir_path: str, extensions):
        """按去扩展名后的基名排序，保证与标签一一对应。"""
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"目录不存在：{dir_path}")
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
        if not paths:
            raise FileNotFoundError(f"目录 {dir_path} 下无{extensions}格式文件")
        paths.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
        return paths

    def _precheck_force_data(self) -> None:
        """对少量样本做快速一致性检查（形状/类型/缺失率/参考数）。"""
        sample_force_paths = self.force_paths[:3]
        for path in sample_force_paths:
            force_data = np.load(path)
            assert force_data.ndim == 2, f"力场文件 {os.path.basename(path)} 格式错误：需2D矩阵"
            assert np.issubdtype(force_data.dtype, np.number), f"力场文件 {os.path.basename(path)} 需为数值类型"
            unique_vals = np.unique(force_data)
            if -1 in unique_vals:
                missing_ratio = (force_data == -1).sum() / (force_data.shape[0] * force_data.shape[1])
                print(f"提前校验：力场文件 {os.path.basename(path)} 缺失率{missing_ratio:.1%}（标识为-1）")
            valid_ref_count = ((force_data != -1) & (force_data != 0)).sum()
            if valid_ref_count == 0:
                raise ValueError(f"力场文件 {os.path.basename(path)} 无有效参考点（需含'非-1且非0'的已知点）")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """返回 (image_tensor[C,H,W], force_tensor[H,W])。"""
        img_path = self.image_paths[idx]
        try:
            with Image.open(img_path) as img:
                # 统一尺寸 + 转 RGB + 归一化
                img_resized = img.resize(self.image_size, Image.BILINEAR)
                img_rgb = img_resized.convert("RGB")
                img_np = np.array(img_rgb, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        except Exception as e:
            raise RuntimeError(f"加载图像 {os.path.basename(img_path)} 失败：{str(e)}")

        force_path = self.force_paths[idx]
        try:
            force_original = np.load(force_path)
            total_pixels = force_original.shape[0] * force_original.shape[1]
            missing_count = (force_original == -1).sum()
            missing_ratio = missing_count / total_pixels
            valid_ref_count = ((force_original != -1) & (force_original != 0)).sum()

            if self.fill_missing and missing_count > 0:
                if missing_ratio > 0.1:
                    print(f"处理力场 {os.path.basename(force_path)}：缺失率{missing_ratio:.1%}，有效参考点{valid_ref_count}个")
                force_filled = fill_missing_force_values(force_original, self.decay_rate)
            else:
                # 若不填补，仅将缺失位置置 0，以确保训练/评估的张量形状与取值可用
                force_filled = np.where(force_original == -1, 0, force_original).astype(np.float32)

            # 尺寸不一致时进行双线性插值到目标大小（H,W）
            if force_filled.shape != self.image_size:
                print(f"调整力场 {os.path.basename(force_path)} 尺寸：{force_filled.shape} → {self.image_size}（双线性插值）")
                force_filled = zoom(
                    force_filled,
                    zoom=(self.image_size[0] / force_filled.shape[0], self.image_size[1] / force_filled.shape[1]),
                    order=1,
                )
            force_tensor = torch.from_numpy(force_filled)
        except Exception as e:
            raise RuntimeError(f"加载力场 {os.path.basename(force_path)} 失败：{str(e)}")

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, force_tensor


def get_dataloaders(fill_missing: bool | None = None, decay_rate: float | None = None):
    """构建训练/测试 DataLoader（可覆盖默认填补策略与衰减率）。

    返回:
        (train_loader, test_loader)
    """
    final_fill_missing = DATA_CONFIG["fill_missing"] if fill_missing is None else fill_missing
    final_decay_rate = DATA_CONFIG["decay_rate"] if decay_rate is None else decay_rate

    train_image_dir = os.path.join(DATA_CONFIG["data_path"], "train", "images")
    train_force_dir = os.path.join(DATA_CONFIG["data_path"], "train", "forces")
    test_image_dir = os.path.join(DATA_CONFIG["data_path"], "test", "images")
    test_force_dir = os.path.join(DATA_CONFIG["data_path"], "test", "forces")

    train_dataset = ForceFieldDataset(
        image_dir=train_image_dir,
        force_dir=train_force_dir,
        image_size=DATA_CONFIG["image_size"],
        fill_missing=final_fill_missing,
        decay_rate=final_decay_rate,
    )
    test_dataset = ForceFieldDataset(
        image_dir=test_image_dir,
        force_dir=test_force_dir,
        image_size=DATA_CONFIG["image_size"],
        fill_missing=final_fill_missing,
        decay_rate=final_decay_rate,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=DATA_CONFIG["batch_size"],
        shuffle=True,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=DATA_CONFIG["batch_size"],
        shuffle=False,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True,
        drop_last=False,
    )

    print("=" * 50)
    print("数据加载器配置完成：")
    print(f"  训练集：{len(train_dataset)}个样本 → {len(train_loader)}个批次")
    print(f"  测试集：{len(test_dataset)}个样本 → {len(test_loader)}个批次")
    print(f"  缺失值填充：{'启用' if final_fill_missing else '禁用'}")
    print(f"  指数衰减率：{final_decay_rate}")
    print(f"  目标图像尺寸：{DATA_CONFIG['image_size']}（H×W）")
    print("=" * 50)
    return train_loader, test_loader


def preprocess_image(image_path: str) -> torch.Tensor:
    """推理阶段的单图预处理：读取 → resize → 归一化 → (1,C,H,W)。"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"预测图像不存在：{image_path}")
    try:
        with Image.open(image_path) as img:
            img_resized = img.resize(DATA_CONFIG["image_size"], Image.BILINEAR)
            img_rgb = img_resized.convert("RGB")
            img_np = np.array(img_rgb, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(BASE_CONFIG["device"])
    except Exception as e:
        raise RuntimeError(f"加载预测图像失败：{str(e)}")
