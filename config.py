"""全局配置（数据、训练、模型、预测）

说明
------
- 本文件用于集中管理工程配置项，不包含业务逻辑。
- 设备 (device) 会在运行时根据可用性自动回退至 CPU。
- 上半部分为 **默认配置模板（注释掉，仅供参考/拷贝）**；
  下半部分为 **Demo 配置（实际生效）**。
"""

import torch

# =============================================================================
#                           DEFAULT CONFIG (TEMPLATE)
#                —— 作为默认参数使用；需要启用时请复制到下方并根据需要修改
# =============================================================================
# # 基础/IO 配置
# BASE_CONFIG = {
#     "seed": 42,
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "output_dir": "./results/",          # 训练日志输出目录
#     "checkpoint_dir": "./checkpoints/",  # 模型权重保存根目录
# }
#
# # 数据配置
# DATA_CONFIG = {
#     "image_size": (1080, 1920),         # (H, W)
#     "num_channels": 3,                  # 输入通道数（RGB=3）
#     "batch_size": 8,                    # 批大小
#     "train_ratio": 0.8,                 # 训练集占比（若使用随机切分）
#     "data_path": "./dataset/",          # 数据集根目录
#     "fill_missing": True,               # 是否填补力场缺失值（值为 -1）
#     "decay_rate": 0.1,                  # 缺失值填补的指数衰减率
#     "missing_value_indicator": -1,      # 缺失值标记
# }
#
# # 模型（PINN）配置
# PINN_CONFIG = {
#     "hidden_dim": 128,                  # MLP 隐藏层维度
#     "num_hidden_layers": 4,             # 隐藏层数量
#     "learning_rate": 1e-3,              # 学习率
#     "train_steps": 20000,               # 训练步数
#     "lambda_data": 1.0,                 # 数据项损失权重
#     "lambda_pde": 0.5,                  # PDE 物理残差损失权重
#     # 可选：若涉及物理网格尺度，建议显式配置像素物理间距
#     # "pixel_size": (Δx, Δy),
#     "physics_params": {                 # 物理参数（按任务需求设置）
#         "E": 1.2e6,                     # 杨氏模量 (Pa)
#         "nu": 0.35,                     # 泊松比
#         "h": 5e-4,                      # 厚度 (m)
#         "k": 1.5e4,                     # 基底刚度 (N/m^3)
#         "EI": 1.0,                      # 抗弯刚度
#         "Kc": 1.0,                      # 弹簧刚度
#         "Gc": 1.0,                      # 剪切刚度
#         "F": 1.0,                       # 轴向载荷（占位）
#         "Fc": 0.8,                      # 临界载荷（占位）
#     },
# }
#
# # 训练/评估配置
# TRAIN_CONFIG = {
#     "log_interval": 100,                # 日志打印步频
#     "save_interval": 1000,              # checkpoint 保存步频
#     "eval_interval": 2000,              # 评估步频（若训练脚本内集成评估）
# }
#
# # 预测/可视化配置
# PREDICT_CONFIG = {
#     "visualize": True,                  # 是否保存可视化图片
#     "save_results": True,               # 是否保存 .npy 等结果文件
# }

# =============================================================================
#                                  DEMO CONFIG
#                           —— 轻量参数，便于快速跑通流程
# =============================================================================

# 基础/IO 配置
BASE_CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "./results/",          # 训练日志输出目录
    "checkpoint_dir": "./checkpoints/",  # 模型权重保存根目录
    # 说明：预测输出目录在脚本中使用 BASE_CONFIG.get("pred_dir", "./predictions") 获取
}

# 数据配置（与 mini_dataset.py 生成的数据一致）
DATA_CONFIG = {
    "image_size": (64, 64),              # (H, W)
    "num_channels": 3,                   # RGB
    "batch_size": 2,
    "train_ratio": 0.8,                  # 预留参数；mini 数据集已固定 train/test
    "data_path": "./mini_dataset",       # Demo 数据根目录
    "fill_missing": True,                # 启用缺失值填补
    "decay_rate": 0.1,                   # 指数衰减率
    "missing_value_indicator": -1,       # 缺失值标记
}

# 模型（PINN）配置（小网络，快速验证）
PINN_CONFIG = {
    "hidden_dim": 32,                    # 较小隐藏层便于快速验证
    "num_hidden_layers": 2,
    "learning_rate": 1e-3,
    "train_steps": 200,                  # Demo：训练步数较少
    "lambda_data": 1.0,
    "lambda_pde": 0.5,
    "pixel_size": (1.0, 1.0),            # 若有物理尺度，请按需修改
    "physics_params": {                  # 物理参数（示例/占位）
        "E": 1.2e6,                      # 杨氏模量 (Pa)
        "nu": 0.35,                      # 泊松比
        "h": 5e-4,                       # 厚度 (m)
        "k": 1.5e4,                      # 基底刚度 (N/m^3)
        "EI": 1.0,                       # 抗弯刚度
        "Kc": 1.0,                       # 弹簧刚度
        "Gc": 1.0,                       # 剪切刚度
        "F": 1.0,                        # 轴向载荷（占位）
        "Fc": 0.8,                       # 临界载荷（占位）
    },
}

# 训练/评估配置
TRAIN_CONFIG = {
    "log_interval": 20,                  # 更密集地打印日志，便于观察收敛
    "save_interval": 50,                 # 在 200 步内会产生 checkpoint_50、_100、_150
    "eval_interval": 100,                # 预留（当前评估由独立脚本执行）
}

# 预测/可视化配置
PREDICT_CONFIG = {
    "visualize": True,                   # 保存预测可视化图像
    "save_results": True,                # 保存 .npy 等结果文件
}
