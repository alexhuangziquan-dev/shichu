"""训练脚本：MechanicsPINN 的端到端训练流程。

要点
- 从首个 batch 动态推断输入/输出维度，构建 MLP。
- 总损失 = λ_data·MSE(预测,真值) + λ_pde·PDE损失（可设为0以关闭）。
- 周期性保存 checkpoint，并保存最终模型。
"""

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
from utils import (
    ensure_dir, make_timestamped_dir, as_device,
    move_batch_to_device, unpack_batch,
    save_checkpoint_train,
)


def setup_dirs():
    """确保结果与权重目录存在，返回 (out_dir, ckpt_root)。"""
    out_dir = ensure_dir(BASE_CONFIG.get("output_dir", "./results"))
    ckpt_root = ensure_dir(os.path.join(BASE_CONFIG.get("ckpt_dir", "./checkpoints"), "pinn"))
    return out_dir, ckpt_root


def setup_logger(out_dir: str) -> None:
    """配置训练日志（文件 + 控制台）。"""
    log_path = os.path.join(out_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def make_unique_ckpt_dir(ckpt_root: str) -> str:
    """在 checkpoints/pinn/ 下创建唯一目录：YYYY_MM_DD_HH_MM_pinn。"""
    run_dir = make_timestamped_dir(ckpt_root, suffix="_pinn")
    return run_dir


def save_checkpoint(ckpt_dir: str, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, extra=None) -> None:
    """保存训练中间权重（与最终权重格式保持一致字段）。"""
    path = os.path.join(ckpt_dir, f"checkpoint_{step}.pth")
    save_checkpoint_train(
        path=path,
        model=model,
        optimizer=optimizer,
        data_config=DATA_CONFIG,
        pinn_config=PINN_CONFIG,
        train_config=TRAIN_CONFIG,
        step=step,
        extra=extra,
    )
    logging.info(f"已保存检查点: {path}")


def train_pinn() -> None:
    """训练主流程。"""
    out_dir, ckpt_root = setup_dirs()
    setup_logger(out_dir)

    ckpt_dir = make_unique_ckpt_dir(ckpt_root)
    logging.info(f"本次训练的 checkpoint 目录：{ckpt_dir}")

    device = as_device(BASE_CONFIG.get("device", "cuda"))
    logging.info(f"使用设备: {device.type if hasattr(device,'type') else device}")

    # 数据加载器
    train_loader, test_loader = get_dataloaders()
    logging.info(f"训练集样本数: {len(train_loader.dataset)}, 测试集样本数: {len(test_loader.dataset)}")

    # 从首个 batch 推断输入/输出维度（避免手工同步配置）
    sample_batch = next(iter(train_loader))
    x_s, y_s, _ = unpack_batch(sample_batch)
    x_s_flat = x_s.reshape(x_s.size(0), -1)
    y_s_flat = y_s.reshape(y_s.size(0), -1)
    in_dim = x_s_flat.size(1)
    out_dim = y_s_flat.size(1)

    hidden_dim = PINN_CONFIG.get("hidden_dim", 128)
    num_hidden = PINN_CONFIG.get("num_hidden_layers", 3)
    layers = [in_dim] + [hidden_dim] * num_hidden + [out_dim]

    # 构建 PINN
    model = MechanicsPINN(
        layers=layers,
        device=device,
        material_params=PINN_CONFIG.get("physics_params", {}),
        image_size=DATA_CONFIG["image_size"],
        pixel_size=PINN_CONFIG.get("pixel_size", (1.0, 1.0)),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=PINN_CONFIG.get("learning_rate", 1e-3))

    total_steps = PINN_CONFIG.get("train_steps", 200)
    log_interval = TRAIN_CONFIG.get("log_interval", 20)
    save_interval = TRAIN_CONFIG.get("save_interval", 100)
    eval_interval = TRAIN_CONFIG.get("eval_interval", 100)  # 预留

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
        x, y, meta = unpack_batch(batch)

        # MLP 输入/输出均为展平向量
        x = x.reshape(x.size(0), -1)
        y = y.reshape(y.size(0), -1)

        # 前向与损失
        y_pred = model(x)
        data_loss = F.mse_loss(y_pred, y)

        if lambda_pde > 0.0:
            # 这里将 y 作为 P（载荷占位），以保持和现有逻辑一致
            lam = {"lambda_pde": lambda_pde, "lambda_bc": 0.0}
            P = y
            _, physics_losses = model.loss_function(x, P, lam, None, None)
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

    # 保存最终权重（带时间戳）
    final_name = f"final_{datetime.now().strftime('%Y_%m_%d_%H_%M')}_pinn.pth"
    final_path = os.path.join(ckpt_dir, final_name)
    save_checkpoint_train(
        path=final_path,
        model=model,
        optimizer=optimizer,
        data_config=DATA_CONFIG,
        pinn_config=PINN_CONFIG,
        train_config=TRAIN_CONFIG,
        step=total_steps,
        extra=None,
    )
    logging.info(f"最终模型已保存到：{final_path}")


if __name__ == "__main__":
    train_pinn()
