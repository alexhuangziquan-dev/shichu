# predict.py
import os
import re
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

from config import BASE_CONFIG, PINN_CONFIG, PREDICT_CONFIG, DATA_CONFIG
from pinn import MechanicsPINN
from data_loader import preprocess_image


# ----------------------------
# 日志
# ----------------------------
def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "prediction.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def as_device():
    want = str(BASE_CONFIG.get("device", "cuda")).lower()
    if want.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------
# 自动解析 checkpoint（与 evaluate.py 保持一致）
# ----------------------------
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


# ----------------------------
# 载入/重建模型
# ----------------------------
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


# ----------------------------
# 可视化
# ----------------------------
def postprocess_force_field(force_tensor: torch.Tensor, image_size: tuple[int, int]) -> np.ndarray:
    H, W = image_size
    return force_tensor.detach().float().cpu().numpy().reshape(H, W)

def visualize_results(input_image, force_pred: np.ndarray, output_path: str, gt_force: np.ndarray | None):
    if gt_force is not None:
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(input_image)
    axes[0].set_title("输入图像")
    axes[0].axis("off")

    im1 = axes[1].imshow(force_pred, cmap="viridis")
    axes[1].set_title("预测力场")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if gt_force is not None:
        im2 = axes[2].imshow(gt_force, cmap="viridis")
        axes[2].set_title("真实力场")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# ----------------------------
# 主函数
# ----------------------------
def predict():
    parser = argparse.ArgumentParser(description="使用 PINN 模型预测力场")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型路径：可为具体文件(.pth)或目录（将自动挑选 final/ckpt）。不传则自动选最新一次训练。")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output", type=str, default="predictions", help="输出目录（默认：predictions）")
    args = parser.parse_args()

    # 新建唯一输出目录
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M")
    out_dir = os.path.join(args.output, "pinn", f"{ts}_pred")
    setup_logging(out_dir)

    # 设备
    device = as_device()
    logging.info(f"使用设备: {device.type}")

    # 加载模型
    model_path = resolve_model_path(args.model_path)
    logging.info(f"加载模型：{model_path}")
    model = build_model_from_ckpt(model_path, device)

    # 预处理输入
    x = preprocess_image(args.input)
    if x.dtype != torch.float32:
        x = x.float()
    x = x.to(device)
    x_flat = x.reshape(1, -1)

    with torch.no_grad():
        y_flat = model(x_flat)[0]

    # 后处理
    H, W = DATA_CONFIG["image_size"]
    y_map = postprocess_force_field(y_flat, (H, W))

    # 保存 .npy
    base = os.path.splitext(os.path.basename(args.input))[0]
    npy_path = os.path.join(out_dir, f"{base}_force.npy")
    np.save(npy_path, y_map)

    # 可视化（预测 vs 真实）
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
        vis_path = os.path.join(out_dir, f"{base}_visualization.png")
        visualize_results(pil_img, y_map, vis_path, gt)
        logging.info(f"可视化已保存: {vis_path}")

    logging.info(f"预测完成，结果保存到：{out_dir}\n- NPY: {npy_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    predict()
