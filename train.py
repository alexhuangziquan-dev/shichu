# train.py
import os
import json
import logging
from datetime import datetime
from itertools import cycle

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from config import DATA_CONFIG, PINN_CONFIG, TRAIN_CONFIG, BASE_CONFIG
from data_loader import get_dataloaders
from pinn import MechanicsPINN


# ----------------------------
# 工具：目录 & 日志
# ----------------------------
def setup_dirs():
    out_dir = BASE_CONFIG.get("output_dir", "./results")
    ckpt_root = os.path.join(BASE_CONFIG.get("ckpt_dir", "./checkpoints"), "pinn")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)
    return out_dir, ckpt_root


def setup_logger(out_dir):
    log_path = os.path.join(out_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ],
        force=True
    )


def make_unique_ckpt_dir(ckpt_root: str) -> str:
    """
    在 checkpoints/pinn/ 下新建唯一目录：YYYY_MM_DD_HH_MM_pinn
    """
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M")
    run_dir = os.path.join(ckpt_root, f"{ts}_pinn")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ----------------------------
# 工具：batch 设备统一 & 取字段
# ----------------------------
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
        x = batch[0]
        y = batch[1]
        x_bc, f_bc = None, None
    elif isinstance(batch, dict):
        x = batch.get("image", None)
        y = batch.get("force", None)
        x_bc = batch.get("x_bc", None)
        f_bc = batch.get("f_bc", None)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")
    return x, y, x_bc, f_bc


# ----------------------------
# 工具：保存 checkpoint
# ----------------------------
def save_checkpoint(ckpt_dir, step, model, optimizer, extra=None):
    path = os.path.join(ckpt_dir, f"checkpoint_{step}.pth")
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "data_config": DATA_CONFIG,
        "pinn_config": PINN_CONFIG,
        "train_config": TRAIN_CONFIG
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    logging.info(f"已保存检查点: {path}")


# ----------------------------
# 主流程
# ----------------------------
def train_pinn():
    # 目录 & 日志
    out_dir, ckpt_root = setup_dirs()
    setup_logger(out_dir)

    # 每次训练新建唯一子目录
    ckpt_dir = make_unique_ckpt_dir(ckpt_root)
    logging.info(f"本次训练的 checkpoint 目录：{ckpt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device.type}")

    # DataLoader
    train_loader, test_loader = get_dataloaders(DATA_CONFIG)
    logging.info(f"训练集样本数: {len(train_loader.dataset)}, 测试集样本数: {len(test_loader.dataset)}")

    # ====== 用首个 batch 动态推断 in_dim/out_dim，再构建模型 ======
    sample_batch = next(iter(train_loader))
    x_s, y_s, _, _ = unpack_batch(sample_batch)

    x_s_flat = x_s.reshape(x_s.size(0), -1)
    y_s_flat = y_s.reshape(y_s.size(0), -1)

    in_dim = x_s_flat.size(1)
    out_dim = y_s_flat.size(1)

    hidden_dim = PINN_CONFIG.get("hidden_dim", 128)
    num_hidden = PINN_CONFIG.get("num_hidden_layers", 3)
    layers = [in_dim] + [hidden_dim] * num_hidden + [out_dim]

    model = MechanicsPINN(
        layers=layers,
        device=device,
        material_params=PINN_CONFIG.get("physics_params", {}),
        image_size=DATA_CONFIG["image_size"],
        pixel_size=PINN_CONFIG.get("pixel_size", (1.0, 1.0))
    ).to(device)

    # 优化器
    lr = PINN_CONFIG.get("learning_rate", 1e-3)
    optimizer = Adam(model.parameters(), lr=lr)

    # 训练配置
    total_steps = PINN_CONFIG.get("train_steps", 200)
    log_interval = TRAIN_CONFIG.get("log_interval", 20)
    save_interval = TRAIN_CONFIG.get("save_interval", 100)
    eval_interval = TRAIN_CONFIG.get("eval_interval", 100)

    lambda_pde = float(PINN_CONFIG.get("lambda_pde", 0.0))
    lambda_data = float(PINN_CONFIG.get("lambda_data", 1.0))
    if lambda_pde == 0.0:
        logging.info("PDE 权重为 0，本轮训练将跳过 PDE 项。")

    train_iter = cycle(train_loader)
    pbar = tqdm(range(total_steps))

    for step in pbar:
        model.train()

        batch = next(train_iter)
        batch = move_batch_to_device(batch, device)
        x, y, x_bc, f_bc = unpack_batch(batch)

        x = x.reshape(x.size(0), -1)
        y = y.reshape(y.size(0), -1)

        # 前向 & data loss
        y_pred = model(x)
        data_loss = F.mse_loss(y_pred, y)

        if lambda_pde > 0.0:
            lam = {"lambda_pde": lambda_pde,
                   "lambda_bc": 0.0 if (x_bc is None or f_bc is None) else 1.0}
            P = y
            _, physics_losses = model.loss_function(x, P, lam, x_bc, f_bc)
            pde_loss = sum(float(v) for v in physics_losses.values())
        else:
            pde_loss = 0.0

        loss = lambda_data * data_loss + lambda_pde * pde_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss={loss.item():.4e} data={data_loss.item():.4e} pde={pde_loss:.4e}")
        if (step + 1) % log_interval == 0 or step == 0:
            logging.info(f"[step {step+1}/{total_steps}] loss={loss.item():.6f}")

        if (step + 1) % save_interval == 0 or (step + 1) == total_steps:
            save_checkpoint(ckpt_dir, step + 1, model, optimizer)

    # 保存最终模型（带时间戳）
    final_name = f"final_{datetime.now().strftime('%Y_%m_%d_%H_%M')}_pinn.pth"
    final_path = os.path.join(ckpt_dir, final_name)
    torch.save({
        "step": total_steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "data_config": DATA_CONFIG,
        "pinn_config": PINN_CONFIG,
        "train_config": TRAIN_CONFIG
    }, final_path)
    logging.info(f"最终模型已保存到：{final_path}")


if __name__ == "__main__":
    train_pinn()
