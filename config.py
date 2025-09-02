import torch
import numpy as np
# 基础配置
# BASE_CONFIG = {
#     "seed": 42,
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "output_dir": "./results/",
#     "checkpoint_dir": "./checkpoints/",
# }
#
# # 数据配置
# DATA_CONFIG = {
#     "image_size": (1080, 1920),  # (H, W) 1920*1080
#     "num_channels": 3,           # RGB通道
#     "batch_size": 8,             # 批次大小
#     "train_ratio": 0.8,          # 训练集比例
#     "data_path": "./dataset/",   # 数据集路径
#     "fill_missing": True,        # 是否自动填充缺失的力场数据
#     "decay_rate": 0.1,           # 指数衰减率，值越大衰减越快
#     "missing_value_indicator": -1  # 缺失值标识
# }
#
# # PINN配置
# PINN_CONFIG = {
#     "hidden_dim": 128,           # 隐藏层维度
#     "num_hidden_layers": 4,      # 隐藏层数量
#     "learning_rate": 1e-3,       # 学习率
#     "train_steps": 20000,        # 训练步数
#     "lambda_data": 1.0,          # 数据损失权重
#     "lambda_pde": 0.5,           # 物理方程损失权重
#     "physics_params": {
#         "E": 1.2e6,              # 弹性模量(Pa)
#         "nu": 0.35,              # 泊松比
#         "h": 5e-4,               # 膜厚度(m)
#         "k": 1.5e4,              # 基底刚度(N/m³)
#         "EI": 1.0,               # 弯曲刚度
#         "Kc": 1.0,               # 基底弹簧刚度
#         "Gc": 1.0,               # 剪切刚度
#         "F": 1.0,                # 轴向载荷
#         "Fc": 0.8,               # 临界载荷
#     }
# }
#
# # 训练与评估配置
# TRAIN_CONFIG = {
#     "log_interval": 100,         # 日志输出间隔
#     "save_interval": 1000,       # 模型保存间隔
#     "eval_interval": 2000,       # 评估间隔
# }
#
# 预测配置
# PREDICT_CONFIG = {
#     "visualize": True,           # 是否可视化结果
#     "save_results": True,        # 是否保存结果
# }

###################################    DEMO config    #######################################

BASE_CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "./results/",
    "checkpoint_dir": "./checkpoints/",
}

DATA_CONFIG = {
    "image_size": (64, 64),      # 先用 64×64
    "num_channels": 3,
    "batch_size": 2,
    "train_ratio": 0.8,
    "data_path": "./mini_dataset",  # 我们马上生成这个小数据集
    "fill_missing": True,
    "decay_rate": 0.1,
    "missing_value_indicator": -1
}

PINN_CONFIG = {
    "hidden_dim": 32,            # 小一点
    "num_hidden_layers": 2,      # 少一点层
    "learning_rate": 1e-3,
    "train_steps": 200,          # 先跑 200 步看看
    "lambda_data": 1.0,
    "lambda_pde": 0.5,
    "pixel_size": (1.0, 1.0),
    "physics_params": {
        "E": 1.2e6,              # 弹性模量(Pa)
        "nu": 0.35,              # 泊松比
        "h": 5e-4,               # 膜厚度(m)
        "k": 1.5e4,              # 基底刚度(N/m³)
        "EI": 1.0,               # 弯曲刚度
        "Kc": 1.0,               # 基底弹簧刚度
        "Gc": 1.0,               # 剪切刚度
        "F": 1.0,                # 轴向载荷
        "Fc": 0.8,               # 临界载荷
    }
}

TRAIN_CONFIG = {
    "log_interval": 20,
    "save_interval": 50,         # 这样 200 步内会产出 checkpoint_50 等
    "eval_interval": 100
}

PREDICT_CONFIG = {
    "visualize": True,           # 是否可视化结果
    "save_results": True,        # 是否保存结果
}