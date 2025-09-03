"""PINN 模型：MLP 主干 + 有限差分物理项（用于 2D 力场/位移预测）。

核心要点
- 输入：展平后的图像向量 (B, C*H*W)
- 输出：展平后的力场向量 (B, H*W)
- 物理项：使用一维差分核卷积实现二阶/四阶导与双拉普拉斯（biharmonic）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MechanicsPINN(nn.Module):
    """基于 MLP 的 PINN，内置有限差分算子。"""

    def __init__(self, layers, device="cpu", material_params=None, image_size=None, pixel_size=(1.0, 1.0)):
        """
        参数:
            layers: MLP 各层尺寸列表，如 [in_dim, hidden, ..., out_dim]
            device: 计算设备
            material_params: 物理参数字典，至少包含 EI/Kc/Gc 等（缺省为1.0）
            image_size: (H, W)，用于将扁平向量还原为 2D 网格
            pixel_size: (Δx, Δy)，物理尺度下的像素间距
        """
        super().__init__()
        self.device = device

        # 物理参数（未提供时默认 1.0）
        mp = material_params or {}
        self.EI = torch.tensor(mp.get("EI", 1.0), device=device)  # 抗弯刚度
        self.Kc = torch.tensor(mp.get("Kc", 1.0), device=device)  # 弹簧项
        self.Gc = torch.tensor(mp.get("Gc", 1.0), device=device)  # 梯度项（拉普拉斯系数）

        # 图像几何参数
        assert image_size is not None, "image_size=(H,W) 必须提供"
        self.image_size = tuple(image_size)  # (H, W)
        self.dx, self.dy = float(pixel_size[0]), float(pixel_size[1])

        # MLP 主干
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                net.append(nn.ReLU())
        self.net = nn.Sequential(*net)

        # 一维有限差分核：2阶 [1,-2,1]；4阶 [1,-4,6,-4,1]
        k2 = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32, device=device)
        k4 = torch.tensor([1.0, -4.0, 6.0, -4.0, 1.0], dtype=torch.float32, device=device)
        self.register_buffer("k2x", k2.view(1, 1, 1, 3))
        self.register_buffer("k2y", k2.view(1, 1, 3, 1))
        self.register_buffer("k4x", k4.view(1, 1, 1, 5))
        self.register_buffer("k4y", k4.view(1, 1, 5, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向： (B, C*H*W) → (B, H*W)。"""
        return self.net(x)

    # ---------- 形状辅助 ----------

    def _flat_to_map(self, f_flat: torch.Tensor) -> torch.Tensor:
        """扁平向量 (B, H*W) → 网格 (B,1,H,W)。"""
        B = f_flat.shape[0]
        H, W = self.image_size
        return f_flat.view(B, 1, H, W)

    def _map_to_flat(self, f_map: torch.Tensor) -> torch.Tensor:
        """网格 (B,1,H,W) → 扁平向量 (B, H*W)。"""
        B, _, H, W = f_map.shape
        return f_map.view(B, H * W)

    # ---------- 有限差分算子 ----------

    def d2(self, f_flat: torch.Tensor):
        """二阶导：返回 d²f/dx² 与 d²f/dy²（扁平格式）。"""
        f_map = self._flat_to_map(f_flat)
        fx2 = F.conv2d(F.pad(f_map, (1, 1, 0, 0), mode="reflect"), self.k2x) / (self.dx ** 2)
        fy2 = F.conv2d(F.pad(f_map, (0, 0, 1, 1), mode="reflect"), self.k2y) / (self.dy ** 2)
        return self._map_to_flat(fx2), self._map_to_flat(fy2)

    def d4(self, f_flat: torch.Tensor):
        """四阶导：返回 d⁴f/dx⁴ 与 d⁴f/dy⁴（扁平格式）。"""
        f_map = self._flat_to_map(f_flat)
        fx4 = F.conv2d(F.pad(f_map, (2, 2, 0, 0), mode="reflect"), self.k4x) / (self.dx ** 4)
        fy4 = F.conv2d(F.pad(f_map, (0, 0, 2, 2), mode="reflect"), self.k4y) / (self.dy ** 4)
        return self._map_to_flat(fx4), self._map_to_flat(fy4)

    # ---------- 物理残差 ----------

    def pure_normal_pressure_equation(self, x_coloc: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """简化板模型在法向压力下的残差（示例）：
        residual = EI * biharmonic(f) + Gc * (f_xx + f_yy) + Kc * f - P
        """
        f = self.net(x_coloc)          # (B, H*W)
        fx2, fy2 = self.d2(f)

        # 拉普拉斯与双拉普拉斯（通过二阶导叠加与再次求导）
        lap = self._flat_to_map(fx2 + fy2)  # (B,1,H,W)
        lap2_x = F.conv2d(F.pad(lap, (1, 1, 0, 0), mode="reflect"), self.k2x) / (self.dx ** 2)
        lap2_y = F.conv2d(F.pad(lap, (0, 0, 1, 1), mode="reflect"), self.k2y) / (self.dy ** 2)
        biharmonic = self._map_to_flat(lap2_x + lap2_y)

        residual = self.EI * biharmonic + self.Gc * (fx2 + fy2) + self.Kc * f - P
        return residual

    def loss_function(self, x_coloc: torch.Tensor, P: torch.Tensor, lam, x_bc=None, f_bc=None):
        """组合 PDE 损失（与可选 BC 损失）。

        参数:
            x_coloc: (B, H*W) 用于计算 PDE 残差的输入（此处直接使用展平后的图像向量）
            P: (B, H*W) 外部载荷或等效项（示例中用 y 作占位）
            lam: dict 或 float。dict 时可包含 {"lambda_pde", "lambda_bc"}。
            x_bc, f_bc: 可选的边界条件监督对 (inputs, targets)。

        返回:
            (loss, {"loss_p": ..., "loss_bc": ...})
        """
        device_ = self.EI.device
        if isinstance(lam, dict):
            w_pde = float(lam.get("lambda_pde", lam.get("pde", 0.0)))
            w_bc = float(lam.get("lambda_bc", lam.get("bc", 0.0)))
        else:
            w_pde, w_bc = float(lam), 0.0

        # PDE 损失
        if w_pde > 0.0:
            res_p = self.pure_normal_pressure_equation(x_coloc, P)
            loss_p = torch.mean(res_p ** 2)
        else:
            loss_p = torch.tensor(0.0, device=device_)

        # 边界条件损失（可选）
        if w_bc > 0.0 and (x_bc is not None) and (f_bc is not None):
            if x_bc.dim() > 2:
                x_bc = x_bc.view(x_bc.size(0), -1)
            if f_bc.dim() > 2:
                f_bc = f_bc.view(f_bc.size(0), -1)
            f_bc_pred = self.net(x_bc)
            loss_bc = F.mse_loss(f_bc_pred, f_bc)
        else:
            loss_bc = torch.tensor(0.0, device=device_)

        loss = w_pde * loss_p + w_bc * loss_bc
        return loss, {"loss_p": float(loss_p.detach()), "loss_bc": float(loss_bc.detach())}
