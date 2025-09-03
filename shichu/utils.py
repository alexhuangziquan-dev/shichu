"""通用工具集合：日志/随机性/路径/权重/指标/可视化/模型重建等。

设计原则
- 汇总跨文件使用的可复用函数，避免重复代码。
- 不引入业务侧强依赖，保持工具函数纯净与稳定。
"""

import os
import sys
import math
import time
import json
import logging
import random
import shutil
import argparse
import re
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境下使用无头后端
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    # 中文字体降级列表，尽量避免中文 glyph 警告
    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Hei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
except Exception:
    plt = None


# ================== 日志 / 随机数 / 设备 ==================

def setup_logger(
    name: str = "PINN",
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    to_file: Optional[str] = None,
) -> logging.Logger:
    """创建并返回一个 logger，避免重复添加 handlers。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(handler)
    logger.propagate = False
    if to_file:
        fh = logging.FileHandler(to_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(fh)
    return logger


def seed_everything(seed: int = 42) -> None:
    """设置 Python / NumPy / PyTorch 的随机种子，确保复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def as_device(prefer: str = "cuda"):
    """根据可用性与偏好返回 torch.device。"""
    want = str(prefer).lower()
    if want.startswith("cuda") and (torch is not None) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu") if torch is not None else "cpu"


# ================== 路径 / 时间戳 ==================

def ensure_dir(path: str) -> str:
    """若目录不存在则创建；返回该路径。"""
    os.makedirs(path, exist_ok=True)
    return path


def timestamp_tag() -> str:
    """返回 YYYY_MM_DD_HH_MM 格式的时间戳字符串。"""
    return datetime.now().strftime("%Y_%m_%d_%H_%M")


def make_timestamped_dir(root: str, suffix: Optional[str] = None) -> str:
    """在 root 下创建 {timestamp}{suffix} 子目录，返回其路径。"""
    ensure_dir(root)
    name = timestamp_tag() + (suffix if suffix else "")
    out_dir = os.path.join(root, name)
    ensure_dir(out_dir)
    return out_dir


def copy_file(src: str, dst_dir: str) -> None:
    """拷贝文件到目标目录（保持原文件名）。"""
    ensure_dir(dst_dir)
    shutil.copy(src, os.path.join(dst_dir, os.path.basename(src)))


# ================== 训练格式的权重保存 ==================

def save_checkpoint_train(
    path: str,
    model: "nn.Module",
    optimizer: Optional["torch.optim.Optimizer"],
    data_config: Dict[str, Any],
    pinn_config: Dict[str, Any],
    train_config: Dict[str, Any],
    step: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """保存训练用 checkpoint，统一字段布局，便于后续重建模型。"""
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "data_config": data_config,
        "pinn_config": pinn_config,
        "train_config": train_config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


# ================== Batch / Tensor ==================

def move_batch_to_device(batch: Any, device: "torch.device") -> Any:
    """递归将 batch 中的张量移动到指定设备。"""
    if torch is None:
        return batch
    if isinstance(batch, Mapping):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(v, device) for v in batch)
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch


def unpack_batch(batch):
    """拆包 batch → (x, y, meta)。支持 (list/tuple) 与 dict。"""
    if isinstance(batch, (list, tuple)):
        x = batch[0]
        y = batch[1]
        meta = {}
    elif isinstance(batch, dict):
        x = batch.get("image", None)
        y = batch.get("force", None)
        meta = {k: v for k, v in batch.items() if k not in ("image", "force")}
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")
    return x, y, meta


def count_parameters(model: "nn.Module") -> int:
    """统计模型可训练参数量。"""
    if torch is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """滑动统计器：维护 sum / count / avg（用于记录标量指标）。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.cnt = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.cnt += n

    @property
    def avg(self) -> float:
        return self.sum / self.cnt if self.cnt > 0 else 0.0


# ================== 图像 I/O / 可视化 ==================

def load_image_rgb(img_path: str, image_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """读取 RGB 图像为 [0,1] float32 数组，可选 resize。"""
    if Image is None:
        raise RuntimeError("需要 Pillow 才能读取图像。")
    img = Image.open(img_path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size[1], image_size[0]))  # 注意 PIL resize 需要 (W, H)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def np_to_torch_image(arr: np.ndarray) -> "torch.Tensor":
    """HxWx3 → 1x3xHxW float32 tensor。"""
    if torch is None:
        raise RuntimeError("需要 PyTorch。")
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def postprocess_force_field(force_tensor: "torch.Tensor", image_size: tuple[int, int]) -> np.ndarray:
    """将扁平力场张量转换为 (H, W) numpy 数组（在 CPU 上）。"""
    H, W = image_size
    return force_tensor.detach().float().cpu().numpy().reshape(H, W)


def save_force_map_as_image(force_map: np.ndarray, out_path: str, vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    """单幅力场图保存（含色标）。"""
    if plt is None:
        raise RuntimeError("需要 matplotlib 保存图片。")
    plt.figure(figsize=(8, 6))
    plt.imshow(force_map, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_side_by_side(
    image_rgb: np.ndarray,
    gt_force: Optional[np.ndarray],
    pred_force: np.ndarray,
    out_path: str,
    titles: Tuple[str, str, str] = ("输入图像", "真实力场", "预测力场"),
) -> None:
    """保存输入/真值/预测 的并排对比图。"""
    if plt is None:
        raise RuntimeError("需要 matplotlib 保存图片。")
    cols = 3 if gt_force is not None else 2
    plt.figure(figsize=(5 * cols, 5))

    ax = plt.subplot(1, cols, 1)
    ax.imshow(image_rgb)
    ax.set_title(titles[0])
    ax.axis("off")

    if gt_force is not None:
        ax = plt.subplot(1, cols, 2)
        im = ax.imshow(gt_force, cmap="viridis")
        ax.set_title(titles[1])
        ax.axis("off")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        idx = 3
    else:
        idx = 2

    ax = plt.subplot(1, cols, idx)
    im = ax.imshow(pred_force, cmap="viridis")
    ax.set_title(titles[-1])
    ax.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def visualize_triplet(pil_img, pred_map: np.ndarray, gt_map: np.ndarray | None, out_png: str) -> None:
    """评估可视化：输入/预测/(真值/绝对误差)。"""
    if plt is None:
        raise RuntimeError("需要 matplotlib 保存图片。")
    H, W = pred_map.shape
    fig, axes = plt.subplots(1, 4 if gt_map is not None else 2, figsize=(24 if gt_map is not None else 12, 6))

    axes[0].imshow(pil_img.resize((W, H)))
    axes[0].set_title("输入图像")
    axes[0].axis("off")

    im1 = axes[1].imshow(pred_map, cmap="viridis")
    axes[1].set_title("预测力场")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if gt_map is not None:
        im2 = axes[2].imshow(gt_map, cmap="viridis")
        axes[2].set_title("真实力场")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        im3 = axes[3].imshow(np.abs(pred_map - gt_map), cmap="magma")
        axes[3].set_title("绝对误差")
        axes[3].axis("off")
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# ================== 指标 / JSON ==================

def mse(a: np.ndarray, b: np.ndarray) -> float:
    """均方误差 (MSE)。"""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """平均绝对误差 (MAE)。"""
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def psnr(a: np.ndarray, b: np.ndarray, data_range: Optional[float] = None) -> float:
    """峰值信噪比 (PSNR, dB)。"""
    if data_range is None:
        data_range = float(np.nanmax(a) - np.nanmin(a) + 1e-12)
    m = mse(a, b)
    if m <= 1e-12:
        return 99.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(m)


def compute_metrics(preds: np.ndarray, gts: np.ndarray, missing_val: float = -1.0) -> Dict[str, float]:
    """在 gts != missing_val 的掩膜下计算指标。"""
    mask = (gts != missing_val)
    diff = np.where(mask, preds - gts, 0.0)
    mse_ = np.sum(diff ** 2) / np.sum(mask)
    rmse_ = np.sqrt(mse_)
    mae_ = np.sum(np.abs(diff)) / np.sum(mask)
    eps = 1e-8
    rel = np.sum(np.abs(diff) / (np.abs(gts) + eps) * mask) / np.sum(mask)
    return {"MSE": float(mse_), "RMSE": float(rmse_), "MAE": float(mae_), "RelErr_%": float(rel * 100.0)}


def try_save_json(obj: Dict[str, Any], path: str) -> None:
    """安全写 JSON（吞掉异常以避免影响主流程）。"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ================== 权重发现 / 模型重建 ==================

RUN_DIR_PATTERN = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_pinn$")


def find_latest_run_dir(ckpt_root: str) -> Optional[str]:
    """返回 checkpoints 根目录下最新的运行目录；若无则 None。"""
    if not os.path.isdir(ckpt_root):
        return None
    subdirs = [d for d in os.listdir(ckpt_root) if RUN_DIR_PATTERN.match(d)]
    if not subdirs:
        return None
    subdirs.sort()
    return os.path.join(ckpt_root, subdirs[-1])


def pick_ckpt_in_dir(run_dir: str) -> Optional[str]:
    """优先选择 final_*.pth；否则选择编号最大的 checkpoint_*.pth。"""
    finals = [f for f in os.listdir(run_dir) if f.startswith("final_") and f.endswith(".pth")]
    if finals:
        finals.sort()
        return os.path.join(run_dir, finals[-1])
    cpts = []
    for f in os.listdir(run_dir):
        m = re.match(r"^checkpoint_(\d+)\.pth$", f)
        if m:
            cpts.append((int(m.group(1)), f))
    if cpts:
        cpts.sort()
        return os.path.join(run_dir, cpts[-1][1])
    return None


def resolve_model_path(model_path_arg: Optional[str], ckpt_root: str) -> str:
    """解析模型路径：优先使用传入路径；否则选择最新一次训练目录中的权重。"""
    if model_path_arg:
        if os.path.isdir(model_path_arg):
            ckpt = pick_ckpt_in_dir(model_path_arg)
            if not ckpt:
                raise FileNotFoundError(f"目录下未找到 final_*.pth 或 checkpoint_*.pth: {model_path_arg}")
            return ckpt
        if os.path.isfile(model_path_arg):
            return model_path_arg
        raise FileNotFoundError(f"未找到指定的模型路径：{model_path_arg}")

    run_dir = find_latest_run_dir(ckpt_root)
    if not run_dir:
        raise FileNotFoundError(f"未在 {ckpt_root} 下找到任何运行目录（形如 YYYY_MM_DD_HH_MM_pinn）")
    ckpt = pick_ckpt_in_dir(run_dir)
    if not ckpt:
        raise FileNotFoundError(f"未在 {run_dir} 下找到 final_*.pth 或 checkpoint_*.pth")
    return ckpt


def extract_state_dict(ckpt: dict) -> dict:
    """从常见 checkpoint 格式中提取 state_dict。"""
    for k in ["model", "model_state_dict", "state_dict"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise KeyError("checkpoint 中未找到可用的 state_dict。")


def infer_layers_from_state_dict_relaxed(state_dict: dict) -> list[int]:
    """从线性层权重近似反推 MLP 层尺寸（作为缺省回退）。"""
    linear_weights = [(k, v) for k, v in state_dict.items() if isinstance(v, torch.Tensor) and v.ndim == 2]
    if not linear_weights:
        raise RuntimeError("state_dict 中未找到二维权重（线性层）。")
    dims = [linear_weights[0][1].shape[1]]
    for _, w in linear_weights:
        out_dim, in_dim = w.shape
        if dims[-1] != in_dim:
            continue
        dims.append(out_dim)
    if len(dims) < 2:
        dims = [linear_weights[0][1].shape[1]] + [w.shape[0] for _, w in linear_weights]
    return dims


def build_model_from_ckpt(model_path: str, device: "torch.device", data_fallback: Dict[str, Any], pinn_fallback: Dict[str, Any]):
    """从 checkpoint 内容或回退配置重建 MechanicsPINN 实例。"""
    from pinn import MechanicsPINN  # 延迟导入避免循环依赖
    ckpt = torch.load(model_path, map_location=device)
    data_cfg = ckpt.get("data_config", None)
    pinn_cfg = ckpt.get("pinn_config", None)

    # 优先使用 checkpoint 中保存的配置（保证网络结构一致）
    if isinstance(data_cfg, dict) and isinstance(pinn_cfg, dict):
        H, W = tuple(data_cfg.get("image_size", data_fallback["image_size"]))
        C = int(data_cfg.get("num_channels", data_fallback["num_channels"]))
        hidden_dim = int(pinn_cfg.get("hidden_dim", pinn_fallback.get("hidden_dim", 128)))
        num_hidden = int(pinn_cfg.get("num_hidden_layers", pinn_fallback.get("num_hidden_layers", 3)))
        layers = [C * H * W] + [hidden_dim] * num_hidden + [H * W]
        image_size = (H, W)
        pixel_size = tuple(pinn_cfg.get("pixel_size", pinn_fallback.get("pixel_size", (1.0, 1.0))))
        physics_params = pinn_cfg.get("physics_params", pinn_fallback.get("physics_params", {}))
    else:
        # 若缺失配置，则尽可能从权重结构反推层级，并使用回退配置
        sd = extract_state_dict(ckpt)
        layers = infer_layers_from_state_dict_relaxed(sd)
        image_size = data_fallback["image_size"]
        pixel_size = pinn_fallback.get("pixel_size", (1.0, 1.0))
        physics_params = pinn_fallback.get("physics_params", {})

    model = MechanicsPINN(
        layers=layers,
        device=device,
        material_params=physics_params,
        image_size=image_size,
        pixel_size=pixel_size,
    ).to(device)

    state_dict = extract_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=False)  # 使用 strict=False 兼容轻微字段差异
    model.eval()
    return model
