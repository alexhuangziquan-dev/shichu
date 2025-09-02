import os, numpy as np
from PIL import Image

root = "./mini_dataset"
splits = [("train", 6), ("test", 2)]
H, W = 64, 64

for split, n in splits:
    img_dir = os.path.join(root, split, "images")
    f_dir  = os.path.join(root, split, "forces")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(f_dir,  exist_ok=True)

    for i in range(n):
        # 合成 RGB 图像（简易渐变+噪声）
        yy, xx = np.mgrid[0:H, 0:W]
        base = (np.sin(xx/6.0) + np.cos(yy/7.0)) * 0.5 + 0.5
        img  = np.stack([
            base,
            np.clip(base + 0.1*np.random.randn(H, W), 0, 1),
            np.clip(1 - base, 0, 1)
        ], axis=-1)
        Image.fromarray((img*255).astype(np.uint8)).save(
            os.path.join(img_dir, f"img_{i:03d}.png"))

        # 合成力场（可视化更明显）：高斯+正弦
        cx, cy = W/2, H/2
        r2 = (xx-cx)**2 + (yy-cy)**2
        force = np.exp(-r2/(2*(W*0.25)**2)) + 0.2*np.sin(xx/4.0)
        force = force.astype(np.float32)

        # 随机抠掉 20% 像素，置为 -1（缺失值）
        mask = np.random.rand(H, W) < 0.2
        force[mask] = -1.0

        np.save(os.path.join(f_dir, f"img_{i:03d}.npy"), force)
print("Mini dataset is ready at ./mini_dataset")
