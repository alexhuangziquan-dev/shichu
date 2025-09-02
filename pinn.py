import torch
import torch.nn as nn
import torch.nn.functional as F

class MechanicsPINN(nn.Module):
    def __init__(self, layers, device='cpu', material_params=None, image_size=None, pixel_size=(1.0, 1.0)):
        super().__init__()
        self.device = device
        mp = material_params or {}
        self.EI = torch.tensor(mp.get('EI', 1.0), device=device)
        self.Kc = torch.tensor(mp.get('Kc', 1.0), device=device)
        self.Gc = torch.tensor(mp.get('Gc', 1.0), device=device)

        # 保存图像尺寸与像素物理尺寸（Δx, Δy）
        assert image_size is not None, "image_size=(H,W) 必须提供"
        self.image_size = tuple(image_size)              # (H, W)
        self.dx, self.dy = float(pixel_size[0]), float(pixel_size[1])

        # MLP 主干
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                net.append(nn.ReLU())
        self.net = nn.Sequential(*net)

        # 一维差分核：二阶 [1, -2, 1]；四阶 [1, -4, 6, -4, 1]
        k2 = torch.tensor([1., -2., 1.], dtype=torch.float32, device=device)
        k4 = torch.tensor([1., -4., 6., -4., 1.], dtype=torch.float32, device=device)
        # x 方向核：水平 (1x3, 1x5)；y 方向核：垂直 (3x1, 5x1)
        self.register_buffer('k2x', k2.view(1,1,1,3))
        self.register_buffer('k2y', k2.view(1,1,3,1))
        self.register_buffer('k4x', k4.view(1,1,1,5))
        self.register_buffer('k4y', k4.view(1,1,5,1))

    def forward(self, x):
        # x: (B, C*H*W) → f_flat: (B, H*W)
        return self.net(x)

    # 形状辅助
    def _flat_to_map(self, f_flat):
        B = f_flat.shape[0]
        H, W = self.image_size
        return f_flat.view(B, 1, H, W)

    def _map_to_flat(self, f_map):
        B, _, H, W = f_map.shape
        return f_map.view(B, H*W)

    # 二阶导：∂²f/∂x² 与 ∂²f/∂y²
    def d2(self, f_flat):
        f_map = self._flat_to_map(f_flat)
        fx2 = F.conv2d(F.pad(f_map, (1,1,0,0), mode='reflect'), self.k2x) / (self.dx ** 2)
        fy2 = F.conv2d(F.pad(f_map, (0,0,1,1), mode='reflect'), self.k2y) / (self.dy ** 2)
        return self._map_to_flat(fx2), self._map_to_flat(fy2)

    # 四阶导：∂⁴f/∂x⁴ 与 ∂⁴f/∂y⁴
    def d4(self, f_flat):
        f_map = self._flat_to_map(f_flat)
        fx4 = F.conv2d(F.pad(f_map, (2,2,0,0), mode='reflect'), self.k4x) / (self.dx ** 4)
        fy4 = F.conv2d(F.pad(f_map, (0,0,2,2), mode='reflect'), self.k4y) / (self.dy ** 4)
        return self._map_to_flat(fx4), self._map_to_flat(fy4)

    # === 物理方程：示例为 “纯法向压力” ===
    # 如果你的理论是 1D 梁：只用 x 向导数（fx2, fx4）
    # 如果是 2D 薄板：常见是双向相加（双拉普拉斯等）
    def pure_normal_pressure_equation(self, x_coloc, P):
        f = self.net(x_coloc)                     # (B, H*W)
        fx2, fy2 = self.d2(f)                     # (B, H*W), (B, H*W)
        fx4, fy4 = self.d4(f)                     # (B, H*W), (B, H*W)

        # ① 1D 梁（仅 x 向）：res = EI * f_xxxx + Gc * f_xx + Kc * f - P
        # residual = self.EI * fx4 + self.Gc * fx2 + self.Kc * f - P

        # ② 2D 薄板（各向同性的简化）：∇⁴ ≈ f_xxxx + 2 f_xx,yy + f_yyyy
        # 这里用各向分离近似：∇² f ≈ fx2 + fy2；∇⁴ f ≈ ∇²(∇² f)
        lap = self._flat_to_map(fx2 + fy2)                           # (B,1,H,W)
        # 对 lap 再做一遍二阶得到双拉普拉斯
        lap2_x = F.conv2d(F.pad(lap, (1,1,0,0), mode='reflect'), self.k2x) / (self.dx ** 2)
        lap2_y = F.conv2d(F.pad(lap, (0,0,1,1), mode='reflect'), self.k2y) / (self.dy ** 2)
        biharmonic = self._map_to_flat(lap2_x + lap2_y)              # (B, H*W)

        residual = self.EI * biharmonic + self.Gc * (fx2 + fy2) + self.Kc * f - P
        return residual

    # 其他物理项（如轴向/总形变）也请改成对 f 的空间差分，保持 (B,H*W)
    # ...

    # 建议：loss_function 里只做 PDE/BC 的组合；data loss 在 train.py 里算
    def loss_function(self, x_coloc, P, lam, x_bc=None, f_bc=None):
        """
        组合 PDE 与 BC 的损失。
        - 当 lam 中 lambda_pde <= 0 时，不计算 PDE；
        - 当 lambda_bc <= 0 或未提供 x_bc/f_bc 时，不计算 BC。
        """
        device_ = self.EI.device  # 与模型参数同设备

        # ---- 权重解析 ----
        if isinstance(lam, dict):
            w_pde = float(lam.get('lambda_pde', lam.get('pde', 0.0)))
            w_bc = float(lam.get('lambda_bc', lam.get('bc', 0.0)))
        else:
            w_pde = float(lam)
            w_bc = 0.0

        # ---- PDE 项 ----
        if w_pde > 0.0:
            # 这里假设你已经把 pure_normal_pressure_equation 改成基于 f(x,y) 的空间差分实现
            # 且 x_coloc、P 都已在训练循环里展平到 (B, H*W)
            res_p = self.pure_normal_pressure_equation(x_coloc, P)
            loss_p = torch.mean(res_p ** 2)
        else:
            loss_p = torch.tensor(0.0, device=device_)

        # ---- BC 项 ----
        # 只有在权重>0 且 同时提供了 x_bc 和 f_bc 时才计算
        if w_bc > 0.0 and (x_bc is not None) and (f_bc is not None):
            # 展平到 (B, -1) 后再过 MLP，保持与主通路一致
            if x_bc.dim() > 2:
                x_bc = x_bc.view(x_bc.size(0), -1)
            if f_bc.dim() > 2:
                f_bc = f_bc.view(f_bc.size(0), -1)
            f_bc_pred = self.net(x_bc)
            loss_bc = F.mse_loss(f_bc_pred, f_bc)
        else:
            loss_bc = torch.tensor(0.0, device=device_)

        # ---- 汇总 ----
        loss = w_pde * loss_p + w_bc * loss_bc
        return loss, {
            "loss_p": float(loss_p.detach()),
            "loss_bc": float(loss_bc.detach())
        }