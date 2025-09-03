"""单张图像预测脚本。

流程
- 解析/自动选择模型权重。
- 图像预处理 → 前向推理 → 保存 .npy 与可选可视化。
"""

import os
import argparse
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


def setup_logging(output_dir: str) -> None:
    """配置文件+控制台日志。"""
    ensure_dir(output_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "prediction.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def predict() -> None:
    """对单张图像进行力场预测并保存结果。"""
    parser = argparse.ArgumentParser(description="使用 PINN 模型预测力场")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径：可为具体文件(.pth)或目录；不传则自动选最新一次训练。")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output", type=str, default="predictions", help="输出目录（默认：predictions）")
    args = parser.parse_args()

    # 唯一输出目录：predictions/pinn/YYYY_MM_DD_HH_MM_pred
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M")
    out_dir = os.path.join(args.output, "pinn", f"{ts}_pred")
    ensure_dir(out_dir)
    setup_logging(out_dir)

    device = as_device(BASE_CONFIG.get("device", "cuda"))
    logging.info(f"使用设备: {device.type if hasattr(device,'type') else device}")

    # 自动解析最新权重或使用指定路径
    ckpt_root = os.path.join(BASE_CONFIG.get("ckpt_dir", "./checkpoints"), "pinn")
    model_path = resolve_model_path(args.model_path, ckpt_root)
    logging.info(f"加载模型：{model_path}")
    model = build_model_from_ckpt(model_path, device, DATA_CONFIG, PINN_CONFIG)

    # 预处理 + 前向推理
    x = preprocess_image(args.input)
    if x.dtype != torch.float32:
        x = x.float()
    x = x.to(device)
    x_flat = x.reshape(1, -1)

    with torch.no_grad():
        y_flat = model(x_flat)[0]

    # 保存结果（.npy）与可视化（.png）
    H, W = DATA_CONFIG["image_size"]
    y_map = postprocess_force_field(y_flat, (H, W))

    base = os.path.splitext(os.path.basename(args.input))[0]
    npy_path = os.path.join(out_dir, f"{base}_force.npy")
    np.save(npy_path, y_map)

    # 若存在 GT 力场，则做对比可视化
    gt_path = args.input.replace("images", "forces").replace(".png", ".npy")
    gt = None
    if os.path.exists(gt_path):
        try:
            gt = np.load(gt_path)
            if gt.shape != (H, W):
                gt = gt.reshape(H, W)
        except Exception as e:
            logging.warning(f"加载真实力场失败（仅绘制预测）：{e}")

    if bool(PREDICT_CONFIG.get("visualize", True)):
        pil_img = Image.open(args.input).convert("RGB").resize((W, H))
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

        vis_path = os.path.join(out_dir, f"{base}_visualization.png")
        plt.tight_layout()
        plt.savefig(vis_path, bbox_inches="tight")
        plt.close()
        logging.info(f"可视化已保存: {vis_path}")

    logging.info(f"预测完成，结果保存到：{out_dir}\n- NPY: {npy_path}")


if __name__ == "__main__":
    predict()
