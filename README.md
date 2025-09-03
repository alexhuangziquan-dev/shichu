# 基于 PINN 的端到端力场预测

[![Project Status](https://img.shields.io/badge/status-active-green.svg)](https://github.com/your-org/pinn-force-prediction)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.10%2B-orange.svg)](https://pytorch.org/)

基于物理信息神经网络（Physics-Informed Neural Network, PINN）的力场预测工具，支持从图像输入预测力场分布，集成训练、评估、批量预测及实时摄像头预测功能。


## 项目简介

本项目用 PINN（Physics-Informed Neural Networks） 从 RGB 图像 端到端预测同分辨率的力场分布。项目包含数据加载、模型训练、单图预测、多图预测、实时预测、模型评估等全流程脚本，并支持在 Windows + CUDA 环境下运行。

**核心功能**：
- 端到端 PINN 模型训练（融合数据损失与物理残差损失）
- 多模式预测：单张图像、批量文件、USB 摄像头实时流
- 自动化评估与可视化（MSE/RMSE 等指标 + 热力图对比）
- 灵活配置系统（数据、模型、训练参数可动态调整）


## 安装指南

### 环境要求
- Python 3.8+
- PyTorch 1.10+（含 CUDA 支持更佳）
- 依赖库：`numpy`, `opencv-python`, `pillow`, `matplotlib`, `tqdm`

### 快速安装
```bash
# 克隆仓库（假设）
git clone https://github.com/your-org/pinn-force-prediction.git
cd pinn-force-prediction

# 安装依赖
pip install -r requirements.txt
# 若需实时摄像头功能
pip install opencv-python
```


## 快速上手

### 1. 生成演示数据集
项目提供轻量数据集生成脚本，用于快速验证流程：
```bash
python shichu/mini_dataset.py
# 生成目录：./mini_dataset/{train,test}/{images,forces}
# - images: 64x64 合成 RGB 图像
# - forces: 对应力场数据（含 ~20% 缺失值，标记为 -1）
```

### 2. 训练模型
使用演示配置训练 PINN 模型：
```bash
python shichu/train.py
# 训练日志：./results/train.log
# 模型权重：./checkpoints/pinn/{时间戳}_pinn/
```

### 3. 评估模型
在测试集上评估最新模型性能：
```bash
python shichu/evaluate.py
# 可选：指定模型路径
# python shichu/evaluate.py --model_path ./checkpoints/pinn/2024_05_01_12_00_pinn/
# 评估结果：./predictions/pinn/{时间戳}_eval/（含指标与可视化）
```

### 4. 预测示例
#### 单张图像预测
```bash
python shichu/predict.py --input ./mini_dataset/test/images/img_000.png
# 输出：./predictions/pinn/{时间戳}_pred/（含 .npy 力场与可视化图）
```

#### 批量文件预测
```bash
python shichu/file_predict.py --input_dir ./mini_dataset/test/images
# 输出：./predictions/pinn/{时间戳}_batch_pred/
```

#### 实时摄像头预测
```bash
python shichu/realtime_predict.py
# 可选：指定摄像头索引（默认 0）
# python shichu/realtime_predict.py --camera_index 1
# 输出：./predictions/pinn/{时间戳}_realtime/（定时保存结果）
```


## 配置说明

核心配置文件为 `shichu/config.py`，支持灵活调整参数。主要配置项分类如下：

### 基础配置（`BASE_CONFIG`）
```python
{
    "seed": 42,                  # 随机种子（保证复现性）
    "device": "cuda" if available else "cpu",  # 计算设备
    "output_dir": "./results/",  # 训练日志目录
    "checkpoint_dir": "./checkpoints/"  # 模型权重根目录
}
```

### 数据配置（`DATA_CONFIG`）
```python
{
    "image_size": (64, 64),      # 输入图像尺寸 (H, W)
    "num_channels": 3,           # 通道数（RGB=3，灰度=1）
    "batch_size": 2,             # 批处理大小
    "data_path": "./mini_dataset",  # 数据集根目录
    "fill_missing": True,        # 是否填补力场缺失值（标记为 -1）
    "decay_rate": 0.1            # 缺失值填补的指数衰减率
}
```

### 模型配置（`PINN_CONFIG`）
```python
{
    "hidden_dim": 32,            # MLP 隐藏层维度
    "num_hidden_layers": 2,      # 隐藏层数量
    "learning_rate": 1e-3,       # 学习率
    "train_steps": 200,          # 训练总步数
    "lambda_data": 1.0,          # 数据损失权重
    "lambda_pde": 0.5,           # PDE 物理残差损失权重
    "pixel_size": (1.0, 1.0),    # 物理尺度像素间距 (Δx, Δy)
    "physics_params": {          # 力学参数（如杨氏模量、泊松比等）
        "E": 1.2e6, "nu": 0.35, "h": 5e-4, ...
    }
}
```

### 训练/预测配置
- `TRAIN_CONFIG`：日志打印间隔、权重保存间隔等
- `PREDICT_CONFIG`：可视化开关、实时预测参数（摄像头索引、保存间隔）等


## API 文档

### 核心模块

#### 1. `pinn.py`：物理信息神经网络核心类
```python
class MechanicsPINN(nn.Module):
    """基于 MLP 的 PINN 模型，内置有限差分算子与物理方程"""
    
    def __init__(self, layers, device="cpu", material_params=None, image_size=None, pixel_size=(1.0, 1.0)):
        """
        参数:
            layers: MLP 层尺寸列表，如 [in_dim, hidden_dim, ..., out_dim]
            device: 计算设备（"cuda" 或 "cpu"）
            material_params: 物理参数字典（含 EI, Kc, Gc 等）
            image_size: (H, W)，用于还原扁平向量为 2D 网格
            pixel_size: (Δx, Δy)，物理尺度下的像素间距
        """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：(B, C*H*W) → (B, H*W)"""
    
    def d2(self, f_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算二阶导数 d²f/dx² 与 d²f/dy²（扁平格式输出）"""
    
    def loss_function(self, x_coloc, P, lam, x_bc=None, f_bc=None) -> Tuple[torch.Tensor, dict]:
        """组合 PDE 残差损失与边界条件损失
        参数:
            x_coloc: 用于计算 PDE 的输入（展平图像向量）
            P: 外部载荷（或等效占位项）
            lam: 损失权重（dict 或 float）
            x_bc, f_bc: 可选边界条件监督数据
        返回:
            总损失与损失分项字典
        """
```

#### 2. `train.py`：训练流程
```python
def train_pinn() -> None:
    """端到端训练主流程：
    - 初始化数据加载器与模型
    - 交替优化数据损失（MSE）与物理残差损失（PDE）
    - 周期性保存权重与日志
    """
```

#### 3. `utils.py`：工具函数
| 函数名 | 功能 |
|--------|------|
| `seed_everything(seed)` | 固定随机种子，保证实验复现性 |
| `as_device(prefer="cuda")` | 自动选择计算设备（优先 GPU） |
| `save_checkpoint_train(...)` | 保存训练状态（模型权重、优化器、配置） |
| `postprocess_force_field(...)` | 将模型输出转换为 (H, W) 力场网格 |
| `visualize_triplet(...)` | 生成输入图像、预测力场、真实力场的对比可视化 |


### 脚本参数说明

#### `evaluate.py`
```bash
python shichu/evaluate.py [--model_path 模型路径] [--num_vis 可视化样例数]
# --model_path: 模型权重文件或目录（默认选最新训练结果）
# --num_vis: 保存的可视化样例数量（默认 4）
```

#### `realtime_predict.py`
```bash
python shichu/realtime_predict.py [--camera_index 摄像头索引] [--save_every 保存间隔(秒)] [--no_window]
# --camera_index: USB 摄像头索引（默认 0）
# --save_every: 定时保存间隔（默认 5 秒）
# --no_window: 关闭实时显示窗口（服务器环境适用）
```


## 项目目录结构
```
shichu/
├── checkpoints/          # 训练权重保存根目录（脚本自动创建子目录）
├── data_creater/         # （如有）自定义数据生成脚本
├── dataset/              # （如有）完整数据集
├── docs/                 # 文档
├── mini_dataset/         # Demo 小数据集（与 DATA_CONFIG 对齐）
├── predictions/          # 预测/评估输出目录（脚本自动创建）
├── results/              # 训练日志与其它结
├── config.py             # 全局配置文件
├── train.py              # 模型训练脚本
├── evaluate.py           # 模型评估脚本
├── predict.py            # 单张图像预测脚本
├── file_predict.py       # 批量文件预测脚本
├── realtime_predict.py   # 实时摄像头预测脚本
├── pinn.py               # PINN 模型定义
├── data_loader.py        # 数据集加载逻辑（隐含）
├── utils.py              # 通用工具函数
└── mini_dataset.py       # 演示数据集生成脚本
```


