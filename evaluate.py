"""评估脚本：在测试集上评估 PINN 模型。

功能要点
- 自动解析最新一次训练的权重（或通过 --model_path 指定）。
- 计算指标：MSE / RMSE / MAE / 相对误差(%)。
- 保存若干可视化样例：输入图、预测力场、真实力场、绝对误差。
"""

import os
import json
import logging
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from config import BASE_CONFIG, DATA_CONFIG, PINN_CONFIG
from data_loader import get_dataloaders
from utils import (
    as_device, ensure_dir, make_timestamped_dir,
    resolve_model_path, build_model_from_ckpt,
    move_batch_to_device, unpack_batch,
    compute_metrics, visualize_triplet,
)


def setup_logging_and_outdir() -> str:
    """创建输出目录并配置日志，返回输出路径。"""
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M")
    out_dir = os.path.join(BASE_CONFIG.get("pred_dir", "./predictions"), "pinn", f"{ts}_eval")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    return out_dir


def main() -> None:
    """在测试集上执行评估流程。"""
    import argparse
    parser = argparse.ArgumentParser(description="评估 PINN 模型")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径：文件(.pth)或目录。不传则自动选最新一次训练。")
    parser.add_argument("--num_vis", type=int, default=4, help="保存可视化样例数量（默认4）")
    args = parser.parse_args()

    out_dir = setup_logging_and_outdir()
    device = as_device(BASE_CONFIG.get("device", "cuda"))
    logging.info(f"使用设备: {device.type if hasattr(device,'type') else device}")

    # 解析权重路径并构建模型
    ckpt_root = os.path.join(BASE_CONFIG.get("ckpt_dir", "./checkpoints"), "pinn")
    model_path = resolve_model_path(args.model_path, ckpt_root)
    logging.info(f"加载模型：{model_path}")
    model = build_model_from_ckpt(model_path, device, DATA_CONFIG, PINN_CONFIG)

    # 数据加载器
    _, test_loader = get_dataloaders()
    H, W = DATA_CONFIG["image_size"]

    preds_all, gts_all, saved = [], [], 0
    missing_indicator = DATA_CONFIG.get("missing_value_indicator", -1)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = move_batch_to_device(batch, device)
            x, y, meta = unpack_batch(batch)

            # 注意：保留 x 的未展平版本用于可视化，避免 reshape 误用
            x_img = x

            # MLP 前向使用展平向量
            x = x.reshape(x.size(0), -1)
            y = y.reshape(y.size(0), -1)
            y_hat = model(x)

            # 聚合到 numpy（B,H,W）
            y_hat_np = y_hat.cpu().numpy().reshape(-1, H, W)
            y_np = y.cpu().numpy().reshape(-1, H, W)
            preds_all.append(y_hat_np)
            gts_all.append(y_np)

            # 保存少量可视化样例
            if saved < args.num_vis:
                try:
                    if isinstance(meta, dict) and "img_path" in meta:
                        pil_img = Image.open(meta["img_path"]).convert("RGB").resize((W, H))
                    else:
                        # 从 x_img 恢复图像进行可视化
                        img_np = (x_img[0].detach().float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_np)

                    out_png = os.path.join(out_dir, f"sample_{i:03d}.png")
                    visualize_triplet(pil_img, y_hat_np[0], y_np[0], out_png)
                    saved += 1
                except Exception as e:
                    logging.warning(f"样例可视化失败（跳过该样例）：{e}")

    # 计算整体指标
    preds_all = np.concatenate(preds_all, axis=0)
    gts_all = np.concatenate(gts_all, axis=0)
    metrics = compute_metrics(preds_all, gts_all, missing_val=missing_indicator)
    logging.info(f"评估结果: {metrics}")

    # 保存指标 JSON
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logging.info(f"评估完成，结果保存在：{out_dir}")


if __name__ == "__main__":
    main()
