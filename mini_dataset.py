"""生成一个用于快速验证流程的演示数据集 (mini dataset)。

目录结构：
./mini_dataset/{train,test}/{images,forces}/
- images: 合成 RGB 图像（平滑背景 + 噪声）
- forces: 合成力场，包含约 20% 的缺失值（值为 -1）
"""

import os
import numpy as np
from PIL import Image

from utils import ensure_dir

root = "./mini_dataset"
splits = [("train", 6), ("test", 2)]
H, W = 64, 64

for split, n in splits:
    img_dir = os.path.join(root, split, "images")
    f_dir = os.path.join(root, split, "forces")
    ensure_dir(img_dir)
    ensure_dir(f_dir)

    for i in range(n):
        # 合成图像：正弦/余弦的平滑基底 + 通道噪声
        yy, xx = np.mgrid[0:H, 0:W]
        base = (np.sin(xx / 6.0) + np.cos(yy / 7.0)) * 0.5 + 0.5
        img = np.stack(
            [
                base,
                np.clip(base + 0.1 * np.random.randn(H, W), 0, 1),
                np.clip(1 - base, 0, 1),
            ],
            axis=-1,
        )
        Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(img_dir, f"img_{i:03d}.png"))

        # 合成力场：高斯峰 + 正弦扰动，并随机置 -1 作为缺失
        cx, cy = W / 2, H / 2
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        force = np.exp(-r2 / (2 * (W * 0.25) ** 2)) + 0.2 * np.sin(xx / 4.0)
        force = force.astype(np.float32)
        mask = np.random.rand(H, W) < 0.2
        force[mask] = -1.0

        np.save(os.path.join(f_dir, f"img_{i:03d}.npy"), force)

print("Mini dataset is ready at ./mini_dataset")
