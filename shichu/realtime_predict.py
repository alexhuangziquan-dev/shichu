"""实时预测脚本：从 USB 摄像头读取图像 → PINN 推理 → 动态显示热力图，并按间隔保存结果

功能
----
- 调用 Windows 下端口 x 的 USB 摄像头（通过 camera index 指定）。
- 将摄像头帧作为模型输入，在线生成预测热力图。
- 支持实时窗口显示（可关闭），并每 N 秒自动保存一次 .npy 和可视化 .png。

依赖
----
- 需安装 opencv-python：`pip install opencv-python`
"""

import os
import time
import logging
from datetime import datetime

import numpy as np
import torch

from config import BASE_CONFIG, DATA_CONFIG, PINN_CONFIG, PREDICT_CONFIG
from utils import (
    ensure_dir, as_device, resolve_model_path, build_model_from_ckpt,
    postprocess_force_field,
)

# OpenCV 是实时采集与窗口显示所需依赖
try:
    import cv2
except Exception as e:
    raise ImportError(
        "缺少依赖：请先安装 opencv-python（pip install opencv-python）。"
    ) from e


def _setup_logging(out_dir: str) -> None:
    """配置文件+控制台日志。"""
    ensure_dir(out_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(out_dir, "realtime.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def _preprocess_frame_bgr(frame_bgr: np.ndarray, target_hw: tuple[int, int]) -> torch.Tensor:
    """将摄像头采集的 BGR 帧预处理为 (1,C,H,W) float32，值域[0,1]。

    说明：
    - OpenCV 读入的帧为 BGR 顺序；需转换为 RGB。
    - 与训练阶段保持一致：resize → 归一化。
    """
    H, W = target_hw
    # resize 到目标尺寸
    resized = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    # BGR → RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # 归一化到 [0, 1]
    arr = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)  # C,H,W
    return torch.from_numpy(arr).unsqueeze(0)  # 1,C,H,W


def _heatmap_from_force(y_map: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    """将 (H,W) 的预测力场转为 BGR 热力图（用于 cv2.imshow）。"""
    # 归一化到 [0, 255]
    y_norm = y_map - np.nanmin(y_map)
    denom = (np.nanmax(y_map) - np.nanmin(y_map) + 1e-12)
    y_norm = (y_norm / denom * 255.0).astype(np.uint8)
    # 放大到显示尺寸，并应用色图
    heat = cv2.resize(y_norm, (out_size[1], out_size[0]), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)  # BGR
    return heat_color


def realtime_predict() -> None:
    """主流程：摄像头 → 预处理 → 推理 → 显示/保存。"""
    import argparse
    parser = argparse.ArgumentParser(description="PINN 实时预测（USB 摄像头）")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径：文件(.pth)/目录；不传则自动选最新一次训练。")
    parser.add_argument("--camera_index", type=int, default=None, help="摄像头索引（Windows 端口 x）")
    parser.add_argument("--save_every", type=int, default=None, help="每 N 秒保存一次预测结果（默认读配置）")
    parser.add_argument("--output", type=str, default="predictions", help="输出根目录（默认：predictions）")
    parser.add_argument("--no_window", action="store_true", help="不显示实时窗口（服务器/无显示环境）")
    args = parser.parse_args()

    # 输出目录：predictions/pinn/YYYY_MM_DD_HH_MM_realtime
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M")
    out_dir = os.path.join(args.output, "pinn", f"{ts}_realtime")
    ensure_dir(out_dir)
    _setup_logging(out_dir)

    device = as_device(BASE_CONFIG.get("device", "cuda"))
    logging.info(f"使用设备: {device.type if hasattr(device,'type') else device}")

    # 解析模型权重
    ckpt_root = os.path.join(BASE_CONFIG.get("ckpt_dir", "./checkpoints"), "pinn")
    model_path = resolve_model_path(args.model_path, ckpt_root)
    logging.info(f"加载模型：{model_path}")
    model = build_model_from_ckpt(model_path, device, DATA_CONFIG, PINN_CONFIG)

    # 读取实时配置
    rt_cfg = PREDICT_CONFIG.get("realtime", {})
    cam_idx = args.camera_index if args.camera_index is not None else int(rt_cfg.get("camera_index", 0))
    save_every = args.save_every if args.save_every is not None else int(rt_cfg.get("save_interval_seconds", 5))
    show_window = False if args.no_window else bool(rt_cfg.get("show_window", True))
    window_name = str(rt_cfg.get("window_name", "PINN RealTime"))

    # 打开摄像头（Windows 建议 CAP_DSHOW）
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头（index={cam_idx}）。")

    H, W = DATA_CONFIG["image_size"]
    last_save_t = 0.0
    saved_count = 0

    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    logging.info("实时预测已启动：按 'q' 或 'Esc' 退出。")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                logging.warning("摄像头读帧失败，尝试继续...")
                continue

            # 预处理 → 推理
            x = _preprocess_frame_bgr(frame_bgr, (H, W)).to(device)
            with torch.no_grad():
                y_flat = model(x.reshape(1, -1))[0]
            y_map = postprocess_force_field(y_flat, (H, W))  # (H,W) numpy

            # 可视化窗口：左侧原图，右侧热力图
            if show_window:
                # 将原始帧缩放到可视化大小
                vis_h = max(480, frame_bgr.shape[0])
                vis_w = int(frame_bgr.shape[1] * (vis_h / frame_bgr.shape[0]))
                frame_disp = cv2.resize(frame_bgr, (vis_w, vis_h), interpolation=cv2.INTER_LINEAR)
                heat_disp = _heatmap_from_force(y_map, (vis_h, vis_w))
                concat = cv2.hconcat([frame_disp, heat_disp])
                cv2.imshow(window_name, concat)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # Esc 或 q
                    break

            # 定时保存
            t_now = time.time()
            if t_now - last_save_t >= save_every:
                tag = datetime.now().strftime("%H%M%S")
                npy_path = os.path.join(out_dir, f"rt_{tag}_{saved_count:04d}.npy")
                png_path = os.path.join(out_dir, f"rt_{tag}_{saved_count:04d}.png")
                np.save(npy_path, y_map)

                # 保存可视化 .png（与窗口一致的热力图）
                heat_for_png = _heatmap_from_force(y_map, (H * 6, W * 6))  # 放大保存
                cv2.imwrite(png_path, heat_for_png)
                logging.info(f"[保存] NPY: {npy_path}  PNG: {png_path}")
                saved_count += 1
                last_save_t = t_now
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        logging.info(f"实时预测结束，共保存 {saved_count} 组结果；输出目录：{out_dir}")


if __name__ == "__main__":
    realtime_predict()
