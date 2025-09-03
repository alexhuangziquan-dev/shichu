"""批量预测脚本：对指定目录内的所有图片做 PINN 推理并保存结果

功能
----
- 将原有单图预测扩展为批量：扫描 target_path 目录下的所有图片文件。
- 预测逻辑与 predict.py 一致：预处理 → 前向 → 保存 .npy，
  若存在同名 GT 力场（forces/*.npy）则一并生成对比可视化。

说明
----
- 默认仅扫描当前目录（不递归）；可在 config 的 PREDICT_CONFIG["file_batch"] 中扩展策略。
"""

import os
import glob
import logging
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from config import BASE_CONFIG, PINN_CONFIG, PREDICT_CONFIG, DATA_CONFIG
from data_loader import preprocess_image
from utils import (
    ensure_dir, as_device,
    resolve_model_path, build_model_from_ckpt,
    postprocess_force_field,
)


def _setup_logging(out_dir: str) -> None:
    """配置文件+控制台日志。"""
    ensure_dir(out_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(out_dir, "file_predict.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _collect_images(root: str, exts: list[str]) -> list[str]:
    """在 root 下按后缀收集图片（不递归）。"""
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, f"*{ext}")))
    files.sort()
    return files


def file_predict() -> None:
    """主流程：批量图片预测并保存结果。"""
    import argparse
    parser = argparse.ArgumentParser(description="PINN 批量图片预测")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径：文件(.pth)/目录；不传则自动选最新一次训练。")
    parser.add_argument("--target_path", type=str, required=True, help="待预测图片所在目录")
    parser.add_argument("--output", type=str, default="predictions", help="输出根目录（默认：predictions）")
    parser.add_argument("--no_vis", action="store_true", help="仅保存 .npy，不生成可视化 .png")
    args = parser.parse_args()

    # 输出目录：predictions/pinn/YYYY_MM_DD_HH_MM_batchpred
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M")
    out_dir = os.path.join(args.output, "pinn", f"{ts}_batchpred")
    ensure_dir(out_dir)
    _setup_logging(out_dir)

    device = as_device(BASE_CONFIG.get("device", "cuda"))
    logging.info(f"使用设备: {device.type if hasattr(device,'type') else device}")

    # 解析模型权重
    ckpt_root = os.path.join(BASE_CONFIG.get("ckpt_dir", "./checkpoints"), "pinn")
    model_path = resolve_model_path(args.model_path, ckpt_root)
    logging.info(f"加载模型：{model_path}")
    model = build_model_from_ckpt(model_path, device, DATA_CONFIG, PINN_CONFIG)

    # 收集图片列表
    fb_cfg = PREDICT_CONFIG.get("file_batch", {})
    exts = fb_cfg.get("image_extensions", [".png", ".jpg", ".jpeg", ".bmp"])
    img_list = _collect_images(args.target_path, exts)
    if not img_list:
        raise FileNotFoundError(f"目录无图片：{args.target_path}")

    H, W = DATA_CONFIG["image_size"]
    do_vis = not args.no_vis and bool(PREDICT_CONFIG.get("visualize", True))
    saved = 0

    for img_path in img_list:
        base = os.path.splitext(os.path.basename(img_path))[0]
        logging.info(f"[预测] {base}")

        # 预处理 + 推理
        x = preprocess_image(img_path).to(device)
        if x.dtype != torch.float32:
            x = x.float()
        with torch.no_grad():
            y_flat = model(x.reshape(1, -1))[0]
        y_map = postprocess_force_field(y_flat, (H, W))  # (H,W) numpy

        # 保存 .npy
        npy_path = os.path.join(out_dir, f"{base}_force.npy")
        np.save(npy_path, y_map)

        # 可视化：若存在 GT 则三图对比，否则两图
        if do_vis:
            try:
                gt_path = img_path.replace("images", "forces").replace(".png", ".npy")
                gt = np.load(gt_path) if os.path.exists(gt_path) else None
                if gt is not None and gt.shape != (H, W):
                    gt = gt.reshape(H, W)

                pil_img = Image.open(img_path).convert("RGB").resize((W, H))
                if gt is not None:
                    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
                else:
                    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

                axes[0].imshow(pil_img)
                axes[0].set_title("输入图像")
                axes[0].axis("off")

                im1 = axes[1].imshow(y_map, cmap="viridis")
                axes[1].set_title("预测力场")
                axes[1].axis("off")
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                if gt is not None:
                    im2 = axes[2].imshow(gt, cmap="viridis")
                    axes[2].set_title("真实力场")
                    axes[2].axis("off")
                    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                png_path = os.path.join(out_dir, f"{base}_visualization.png")
                plt.tight_layout()
                plt.savefig(png_path, bbox_inches="tight")
                plt.close()
            except Exception as e:
                logging.warning(f"可视化失败（仅保存 NPY）：{e}")

        saved += 1

    logging.info(f"批量预测完成，处理 {saved} 张图片。输出目录：{out_dir}")


if __name__ == "__main__":
    file_predict()
