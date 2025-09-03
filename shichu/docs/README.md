# 基于 PINN 的端到端力场预测
[![Project Status](https://img.shields.io/badge/status-active-green.svg)](https://github.com/your-org/pinn-force-prediction)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.10%2B-orange.svg)](https://pytorch.org/)
基于物理信息神经网络（Physics-Informed Neural Network, PINN）的力场预测工具，支持从图像输入预测力场分布，集成训练、评估、批量预测及实时摄像头预测等功能。

## 项目简介
本项目用 PINN（Physics-Informed Neural Networks）从 RGB 图像端到端预测同分辨率的力场分布。项目包含数据加载、模型训练、单图预测、多图预测、实时预测、模型评估等全流程脚本，并支持在 Windows + CUDA 环境下运行。

**核心功能**：
- 端到端 PINN 模型训练（融合数据损失与物理残差损失）
- 多模式预测：单张图像、批量文件、USB 摄像头实时流
- 自动化评估与可视化（MSE/RMSE 等指标 + 热力图对比）
- 灵活配置系统（数据、模型、训练参数可动态调整）


## 项目目录结构
```
shichu/
├── checkpoints/          # 模型权重保存根目录（脚本自动创建子目录，按时间戳命名）
├── data_creater/         # （预留）自定义数据生成脚本目录（用户可自行添加）
├── dataset/              # 正式数据集目录（用户需自行准备，结构与 mini_dataset 一致）
├── docs/                 # 项目文档目录（可放入技术报告、API 详细说明等）
├── mini_dataset/         # 演示数据集（脚本生成，用于快速验证流程，正式训练无需依赖）
│   ├── train/
│   │   ├── images/
│   │   └── forces/
│   └── test/
│       ├── images/
│       └── forces/
├── predictions/          # 预测/评估结果输出目录（脚本自动创建，按任务类型+时间戳命名）
├── results/              # 训练日志目录（保存 train.log 及训练过程中的损失曲线等）
├── config.py             # 全局配置文件（数据、模型、训练参数均在此修改）
├── train.py              # 模型训练脚本（入口函数 train_pinn()）
├── evaluate.py           # 模型评估脚本（计算测试集指标并生成可视化）
├── predict.py            # 单张图像预测脚本（支持指定单图路径）
├── file_predict.py       # 批量文件预测脚本（支持指定图像文件夹）
├── realtime_predict.py   # 实时摄像头预测脚本（支持 USB 摄像头输入）
├── pinn.py               # PINN 核心模型定义（MechanicsPINN 类）
├── data_loader.py        # 数据集加载逻辑（自动读取 DATA_CONFIG 指定路径的图像与力场）
├── utils.py              # 通用工具函数（随机种子、设备选择、可视化等）
└── mini_dataset.py       # 演示数据集生成脚本（生成 mini_dataset 目录）
```

## 安装指南
### 环境要求
- Python 3.8+
- PyTorch 1.10+（含 CUDA 支持更佳）
- 依赖库：`numpy`, `opencv-python`, `pillow`, `matplotlib`, `tqdm`

### 快速安装
```bash
# 克隆仓库
git clone https://github.com/alexhuangziquan-dev/shichu.git
cd shichu
# 安装依赖
pip install -r requirements.txt
# 若需实时摄像头功能
pip install opencv-python
```

## 数据集准备
项目提供 `mini_dataset` 用于快速验证流程，**正式使用时需用户自行准备数据集并放入 `dataset` 目录**，其结构与数据格式需与 `mini_dataset` 保持一致，确保模型可正常加载与训练。

### 1. 正式数据集目录结构
需在项目根目录手动创建 `dataset` 文件夹，并按以下层级组织训练/测试数据：
```
dataset/
├── train/                # 训练集
│   ├── images/           # 训练集 RGB 图像
│   │   ├── img_001.png   # 图像文件（命名格式不限，建议有序）
│   │   ├── img_002.png
│   │   └── ...
│   └── forces/           # 训练集对应力场数据
│       ├── img_001.npy   # 力场文件（需与 images 目录文件一一对应，命名一致）
│       ├── img_002.npy
│       └── ...
└── test/                 # 测试集（结构与 train 完全一致）
    ├── images/
    │   ├── img_101.png
    │   └── ...
    └── forces/
        ├── img_101.npy
        └── ...
```

### 2. 数据格式要求
需严格遵循以下格式规范，避免数据加载错误：
| 数据类型 | 规格要求 | 说明 |
|----------|----------|------|
| RGB 图像 | 尺寸：与 `config.py` 中 `DATA_CONFIG["image_size"]` 一致（默认 64x64）<br>通道：3 通道（RGB）<br>格式：PNG（推荐，其他主流格式如 JPG 可兼容） | 图像内容需与力场数据语义对应（如材料表面图像对应其受力分布） |
| 力场数据 | 维度：(H, W)（与图像尺寸一致，H=高度，W=宽度）<br>数据类型：float32<br>缺失值标记：-1（若存在部分受力数据缺失，需用 -1 标注，模型会自动填补）<br>格式：NumPy 二进制文件（.npy） | 力场数据需为物理实测或仿真得到的真实值，确保训练有效性 |

### 3. 数据集配置调整
准备好 `dataset` 后，需修改核心配置文件 `shichu/config.py` 中的 `DATA_CONFIG`，将数据路径指向正式数据集：
```python
# DATA_CONFIG 配置修改示例
DATA_CONFIG = {
    "image_size": (64, 64),      # 需与正式数据集图像尺寸保持一致
    "num_channels": 3,           # 固定为 3（RGB 图像）
    "batch_size": 2,             # 可根据 GPU 显存调整
    "data_path": "./dataset",    # 改为正式数据集根目录
    "fill_missing": True,        # 启用缺失值填补（若力场无缺失可设为 False）
    "decay_rate": 0.1            # 缺失值填补的指数衰减率（无需修改）
}
```

## 快速上手
### 1. 生成演示数据集（可选）
若仅需验证流程，可生成轻量演示数据集 `mini_dataset`；若已准备好 `dataset` 正式数据，可跳过此步：
```bash
python shichu/mini_dataset.py
# 生成目录：./mini_dataset/{train,test}/{images,forces}
# - images: 64x64 合成 RGB 图像
# - forces: 对应力场数据（含 ~20% 缺失值，标记为 -1）
```

### 2. 训练模型
使用配置文件指定的数据集（`mini_dataset` 或 `dataset`）训练 PINN 模型：
```bash
python shichu/train.py
# 训练日志：./results/train.log
# 模型权重：./checkpoints/pinn/{时间戳}_pinn/（按训练开始时间命名）
```

### 3. 评估模型
在测试集上评估最新模型性能（默认使用 `checkpoints` 中最新训练的模型）：
```bash
python shichu/evaluate.py
# 可选：指定特定模型路径（适用于复现或对比不同训练结果）
# python shichu/evaluate.py --model_path ./checkpoints/pinn/2024_05_01_12_00_pinn/
# 评估结果输出：./predictions/pinn/{时间戳}_eval/
# - 包含 MSE/RMSE 等量化指标（.txt）
# - 输入图像、预测力场、真实力场的对比热力图（.png）
```

### 4. 预测示例
#### 单张图像预测
输入单张 PNG 图像，输出对应的力场预测结果：
```bash
python shichu/predict.py --input ./dataset/test/images/img_101.png  # 正式数据集示例
# 或使用演示数据集：python shichu/predict.py --input ./mini_dataset/test/images/img_000.png
# 输出目录：./predictions/pinn/{时间戳}_pred/
# - 预测力场数据（.npy）
# - 输入图像与预测力场的可视化图（.png）
```

#### 批量文件预测
对文件夹内所有图像批量预测力场：
```bash
python shichu/file_predict.py --input_dir ./dataset/test/images  # 正式数据集示例
# 或使用演示数据集：python shichu/file_predict.py --input_dir ./mini_dataset/test/images
# 输出目录：./predictions/pinn/{时间戳}_batch_pred/（含所有图像的预测结果）
```

#### 实时摄像头预测
通过 USB 摄像头实时采集图像并预测力场，定时保存结果：
```bash
python shichu/realtime_predict.py
# 可选参数：
# --camera_index 1：指定摄像头索引（多摄像头时使用，默认 0）
# --save_every 10：设置结果保存间隔（默认 5 秒，单位：秒）
# --no_window：关闭实时显示窗口（适用于服务器无 GUI 环境）
# 输出目录：./predictions/pinn/{时间戳}_realtime/（定时保存实时预测的图像与力场）
```

## 配置说明
核心配置文件为 `shichu/config.py`，支持灵活调整参数。主要配置项分类如下：

### 基础配置（`BASE_CONFIG`）
```python
{
    "seed": 42,                  # 随机种子（保证实验复现性，建议不修改）
    "device": "cuda" if available else "cpu",  # 自动选择计算设备（优先 GPU）
    "output_dir": "./results/",  # 训练日志与中间结果保存目录
    "checkpoint_dir": "./checkpoints/"  # 模型权重保存根目录
}
```

### 数据配置（`DATA_CONFIG`）
详见「数据集准备 - 3. 数据集配置调整」，关键参数已说明。

### 模型配置（`PINN_CONFIG`）
```python
{
    "hidden_dim": 32,            # MLP 隐藏层维度（可调整，如 64/128，需匹配 GPU 显存）
    "num_hidden_layers": 2,      # 隐藏层数量（建议 2-4 层，过多易过拟合）
    "learning_rate": 1e-3,       # 学习率（可根据训练收敛情况调整，如 5e-4/2e-3）
    "train_steps": 200,          # 训练总步数（正式训练建议增至 1000+，视数据量调整）
    "lambda_data": 1.0,          # 数据损失（MSE）权重
    "lambda_pde": 0.5,           # PDE 物理残差损失权重（平衡数据拟合与物理规律）
    "pixel_size": (1.0, 1.0),    # 物理尺度像素间距 (Δx, Δy)，需根据实际场景校准
    "physics_params": {          # 力学参数（需根据具体材料/场景修改）
        "E": 1.2e6, "nu": 0.35, "h": 5e-4, ...  # E=杨氏模量，nu=泊松比，h=厚度等
    }
}
```

### 训练/预测配置
- `TRAIN_CONFIG`：日志打印间隔（如每 10 步打印一次损失）、权重保存间隔（如每 50 步保存一次）等。
- `PREDICT_CONFIG`：可视化开关（是否生成热力图）、实时预测参数（摄像头帧率、保存间隔）等。

## API 文档
### 核心模块
#### 1. `pinn.py`：物理信息神经网络核心类
```python
class MechanicsPINN(nn.Module):
    """基于 MLP 的 PINN 模型，内置有限差分算子与物理方程（如弹性力学 PDE）"""
    
    def __init__(self, layers, device="cpu", material_params=None, image_size=None, pixel_size=(1.0, 1.0)):
        """
        参数说明:
            layers: MLP 层尺寸列表，格式为 [输入维度, 隐藏层维度, ..., 输出维度]
                    （输入维度 = 3*H*W，输出维度 = H*W，H/W 为图像尺寸）
            device: 计算设备（"cuda" 或 "cpu"）
            material_params: 材料力学参数字典（对应 PINN_CONFIG["physics_params"]）
            image_size: (H, W)，用于将模型输出的扁平向量还原为 2D 力场网格
            pixel_size: (Δx, Δy)，物理尺度下的像素间距（用于计算 PDE 导数）
        """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：将展平的图像向量（B, C*H*W）映射为展平的力场向量（B, H*W）"""
    
    def d2(self, f_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算力场的二阶偏导数 d²f/dx² 与 d²f/dy²（输出为扁平格式，后续会还原为 2D）"""
    
    def loss_function(self, x_coloc, P, lam, x_bc=None, f_bc=None) -> Tuple[torch.Tensor, dict]:
        """组合损失函数：PDE 物理残差损失 + 数据监督损失（可选边界条件损失）
        参数:
            x_coloc: 用于计算 PDE 的输入样本（展平图像向量）
            P: 外部载荷数据（或等效占位张量）
            lam: 损失权重字典（含 "data" 和 "pde" 键，对应 lambda_data 和 lambda_pde）
            x_bc, f_bc: 可选边界条件数据（x_bc=边界处输入，f_bc=边界处真实力场）
        返回:
            总损失张量 + 损失分项字典（便于日志跟踪数据损失、PDE 损失各自变化）
        """
```

#### 2. `train.py`：训练流程
```python
def train_pinn() -> None:
    """端到端训练主流程：
    1. 加载配置（从 config.py 读取参数）
    2. 初始化数据加载器（根据 DATA_CONFIG["data_path"] 加载数据集）
    3. 初始化 MechanicsPINN 模型、优化器（Adam 优化器）
    4. 迭代训练：交替计算数据损失与 PDE 残差损失，反向传播更新参数
    5. 周期性操作：保存模型权重（到 checkpoint_dir）、记录训练日志（到 output_dir）
    """
```

#### 3. `utils.py`：工具函数
| 函数名 | 功能 |
|--------|------|
| `seed_everything(seed)` | 固定 Python、NumPy、PyTorch 的随机种子，确保实验结果可复现 |
| `as_device(prefer="cuda")` | 自动检测并返回可用计算设备（优先使用 CUDA，无 GPU 时使用 CPU） |
| `save_checkpoint_train(model, optimizer, epoch, config, save_path)` | 保存训练状态：模型权重、优化器参数、当前 epoch、配置信息（便于断点续训） |
| `postprocess_force_field(f_flat, image_size)` | 将模型输出的扁平力场向量（B, H*W）还原为 (B, H, W) 网格，并处理缺失值 |
| `visualize_triplet(img, pred_force, true_force, save_path)` | 生成三图对比可视化：输入 RGB 图像 + 预测力场热力图 + 真实力场热力图（便于评估） |

### 脚本参数说明
#### `evaluate.py` 命令行参数
```bash
python shichu/evaluate.py [--model_path 模型路径] [--num_vis 可视化样例数]
# 可选参数说明：
# --model_path: 模型权重目录路径（默认自动查找 checkpoints/pinn 下最新的训练结果）
# --num_vis: 保存的可视化样例数量（默认 4，即从测试集中随机选 4 个样本生成对比图）
```

#### `realtime_predict.py` 命令行参数
```bash
python shichu/realtime_predict.py [--camera_index 摄像头索引] [--save_every 保存间隔] [--no_window]
# 可选参数说明：
# --camera_index: USB 摄像头索引（多摄像头设备时指定，默认 0，即第一个摄像头）
# --save_every: 预测结果保存间隔（单位：秒，默认 5，即每 5 秒保存一次当前预测结果）
# --no_window: 仅保存结果不显示实时窗口（适用于 Linux 服务器等无 GUI 环境，无参数值）
```

