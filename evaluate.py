# evaluate.py
import os
import re
import json
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from config import BASE_CONFIG, DATA_CONFIG, PINN_CONFIG
from data_loader import get_dataloaders
from pinn import MechanicsPINN


# =========================
# 日志与目录
# =========================
def setup_logging_and_outdir():
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


def as_device():
    want = str(BASE_CONFIG.get("device", "cuda")).lower()
    if want.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =========================
# 自动解析 checkpoint（与 predict.py 一致）
# =========================
RUN_DIR_PATTERN = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_pinn$")

def find_latest_run_dir(ckpt_root: str) -> str | None:
    if not os.path.isdir(ckpt_root):
        return None
    subdirs = [d for d in os.listdir(ckpt_root) if RUN_DIR_PATTERN.match(d)]
    if not subdirs:
        return None
    subdirs.sort()
    return os.path.join(ckpt_root, subdirs[-1])

def pick_ckpt_in_dir(run_dir: str) -> str | None:
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

def resolve_model_path(model_path_arg: str | None) -> str:
    ckpt_root = os.path.join(BASE_CONFIG.get("ckpt_dir", "./checkpoints"), "pinn")
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
    logging.info(f"自动选择模型：{ckpt}")
    return ckpt


# =========================
# 构建与载入模型
# =========================
def extract_state_dict(ckpt: dict) -> dict:
    for k in ["model", "model_state_dict", "state_dict"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise KeyError("checkpoint 中未找到可用的 state_dict。")

def infer_layers_from_state_dict_relaxed(state_dict: dict) -> list[int]:
    linear_weights = [(k, v) for k, v in state_dict.items()
                      if isinstance(v, torch.Tensor) and v.ndim == 2]
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

def build_model_from_ckpt(model_path: str, device: torch.device) -> MechanicsPINN:
    ckpt = torch.load(model_path, map_location=device)
    data_cfg = ckpt.get("data_config", None)
    pinn_cfg = ckpt.get("pinn_config", None)
    if isinstance(data_cfg, dict) and isinstance(pinn_cfg, dict):
        H, W = tuple(data_cfg.get("image_size", DATA_CONFIG["image_size"]))
        C = int(data_cfg.get("num_channels", DATA_CONFIG["num_channels"]))
        in_dim = C * H * W
        out_dim = H * W
        hidden_dim = int(pinn_cfg.get("hidden_dim", PINN_CONFIG.get("hidden_dim", 128)))
        num_hidden = int(pinn_cfg.get("num_hidden_layers", PINN_CONFIG.get("num_hidden_layers", 3)))
        layers = [in_dim] + [hidden_dim] * num_hidden + [out_dim]
        image_size = (H, W)
        pixel_size = tuple(pinn_cfg.get("pixel_size", PINN_CONFIG.get("pixel_size", (1.0, 1.0))))
        physics_params = pinn_cfg.get("physics_params", PINN_CONFIG.get("physics_params", {}))
        logging.info(f"根据 checkpoint 配置重建网络结构：{layers}")
    else:
        sd = extract_state_dict(ckpt)
        layers = infer_layers_from_state_dict_relaxed(sd)
        image_size = DATA_CONFIG["image_size"]
        pixel_size = PINN_CONFIG.get("pixel_size", (1.0, 1.0))
        physics_params = PINN_CONFIG.get("physics_params", {})
        logging.info(f"从 state_dict 推断网络结构：{layers}")

    model = MechanicsPINN(
        layers=layers,
        device=device,
        material_params=physics_params,
        image_size=image_size,
        pixel_size=pixel_size
    ).to(device)

    state_dict = extract_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# =========================
# 工具函数
# =========================
def move_batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [(t.to(device, non_blocking=True) if torch.is_tensor(t) else t)
                for t in batch]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

def unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        x = batch[0]; y = batch[1]; meta = {}
    elif isinstance(batch, dict):
        x = batch.get("image", None); y = batch.get("force", None)
        meta = {k: v for k, v in batch.items() if k not in ("image", "force")}
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")
    return x, y, meta


def compute_metrics(preds: np.ndarray, gts: np.ndarray, missing_val: float = -1.0):
    mask = (gts != missing_val)
    diff = np.where(mask, preds - gts, 0.0)
    mse = np.sum(diff ** 2) / np.sum(mask)
    rmse = np.sqrt(mse)
    mae = np.sum(np.abs(diff)) / np.sum(mask)
    eps = 1e-8
    rel = np.sum(np.abs(diff) / (np.abs(gts) + eps) * mask) / np.sum(mask)
    return {"MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae), "RelErr_%": float(rel * 100.0)}


def visualize_triplet(pil_img, pred_map, gt_map, out_png):
    H, W = pred_map.shape
    fig, axes = plt.subplots(1, 4 if gt_map is not None else 2, figsize=(24 if gt_map is not None else 12, 6))

    axes[0].imshow(pil_img.resize((W, H)))
    axes[0].set_title("输入图像"); axes[0].axis("off")

    im1 = axes[1].imshow(pred_map, cmap="viridis"); axes[1].set_title("预测力场"); axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if gt_map is not None:
        im2 = axes[2].imshow(gt_map, cmap="viridis"); axes[2].set_title("真实力场"); axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        im3 = axes[3].imshow(np.abs(pred_map - gt_map), cmap="magma"); axes[3].set_title("绝对误差"); axes[3].axis("off")
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close()


# =========================
# 主函数
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="评估 PINN 模型")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型路径：可为具体文件(.pth)或目录。不传则自动选最新一次训练。")
    parser.add_argument("--num_vis", type=int, default=4, help="保存可视化样例数量（默认4）")
    args = parser.parse_args()

    out_dir = setup_logging_and_outdir()
    device = as_device()
    logging.info(f"使用设备: {device.type}")

    model_path = resolve_model_path(args.model_path)
    logging.info(f"加载模型：{model_path}")
    model = build_model_from_ckpt(model_path, device)

    _, test_loader = get_dataloaders(DATA_CONFIG)
    H, W = DATA_CONFIG["image_size"]

    preds_all, gts_all, saved = [], [], 0
    missing_indicator = DATA_CONFIG.get("missing_value_indicator", -1)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = move_batch_to_device(batch, device)
            x, y, meta = unpack_batch(batch)
            x = x.reshape(x.size(0), -1); y = y.reshape(y.size(0), -1)
            y_hat = model(x)

            y_hat_np = y_hat.cpu().numpy().reshape(-1, H, W)
            y_np = y.cpu().numpy().reshape(-1, H, W)
            preds_all.append(y_hat_np); gts_all.append(y_np)

            if saved < args.num_vis:
                if isinstance(meta, dict) and "img_path" in meta:
                    pil_img = Image.open(meta["img_path"]).convert("RGB").resize((W, H))
                else:
                    pil_img = Image.fromarray((x[0].cpu().numpy().reshape(H, W) * 255).astype(np.uint8))
                visualize_triplet(pil_img, y_hat_np[0], y_np[0], os.path.join(out_dir, f"sample_{i:03d}.png"))
                saved += 1

    preds_all = np.concatenate(preds_all, axis=0)
    gts_all = np.concatenate(gts_all, axis=0)
    metrics = compute_metrics(preds_all, gts_all, missing_val=missing_indicator)
    logging.info(f"评估结果: {metrics}")

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logging.info(f"评估完成，结果保存在：{out_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
