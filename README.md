
---

# PINN-ForceField from Images

## 📖 项目简介

本项目实现了一个 **基于物理信息神经网络（Physics-Informed Neural Network, PINN）** 的力场预测框架。
系统能够从输入的 **RGB 图像** 中预测对应的 **二维力场分布**，结合 **数据驱动的学习** 与 **物理约束（偏微分方程残差）**，使预测结果在数值上更符合物理规律。

---

## 📂 项目结构

```bash
shichu/
├── config.py          # 全局配置文件
├── data_loader.py     # 数据集定义 & DataLoader 封装
├── mini_dataset.py    # 生成演示用数据集
├── pinn.py            # PINN 模型定义（MLP 主干 + PDE 残差）
├── train.py           # 训练脚本
├── evaluate.py        # 模型评估脚本
├── predict.py         # 单张图像预测脚本
├── utils.py           # 工具函数库（日志、保存、可视化等）
├── checkpoints/       # 模型权重保存目录
├── results/           # 训练日志保存目录
└── predictions/       # 预测与评估结果保存目录
```

---

## ⚙️ 环境配置

推荐使用 Conda 创建独立环境：

```bash
conda create -n pinn-env python=3.10 -y
conda activate pinn-env
```

安装依赖：

```bash
pip install torch torchvision torchaudio
pip install numpy scipy pillow matplotlib tqdm
```

---

## 🚀 快速开始

### 1. 准备数据

运行以下命令生成一个演示数据集：

```bash
python mini_dataset.py
```

数据集目录结构如下：

```
dataset/
├── train/
│   ├── images/   # 训练图像
│   └── forces/   # 训练力场（.npy）
└── test/
    ├── images/   # 测试图像
    └── forces/   # 测试力场（.npy）
```

### 2. 训练模型

```bash
python train.py
```

* 训练日志将保存到 `results/`
* 模型权重将保存到 `checkpoints/pinn/` 下的时间戳目录

### 3. 模型评估

```bash
python evaluate.py
```

* 自动加载最新训练好的模型
* 输出评估指标（MSE、RMSE、MAE、相对误差%）
* 保存预测结果与可视化对比图到 `predictions/`

### 4. 单张图像预测

```bash
python predict.py --input ./mini_dataset/test/images/img_001.png
```

* 输出 `.npy` 力场文件
* 生成 `.png` 可视化结果

---

## ⚙️ 配置说明

配置集中在 `config.py`，主要参数如下：

### 基础配置

* `seed`：随机种子
* `device`：运行设备（默认 `cuda`，不可用时自动切换到 `cpu`）
* `output_dir`：训练日志保存目录
* `checkpoint_dir`：权重保存目录
* `pred_dir`：预测结果保存目录

### 数据配置

* `image_size`：输入图像尺寸 (H, W)
* `num_channels`：图像通道数（RGB=3）
* `batch_size`：批大小
* `data_path`：数据根目录
* `fill_missing`：是否填补力场缺失值
* `decay_rate`：缺失值填补的指数衰减率
* `missing_value_indicator`：缺失值标记

### 模型配置

* `hidden_dim`：MLP 每层隐藏单元数
* `num_hidden_layers`：隐藏层数量
* `learning_rate`：学习率
* `train_steps`：训练步数
* `lambda_data`：数据损失权重
* `lambda_pde`：PDE 损失权重
* `pixel_size`：像素物理间距 (Δx, Δy)
* `physics_params`：物理参数（EI, Kc, Gc 等）

### 训练配置

* `log_interval`：日志打印间隔
* `save_interval`：模型保存间隔
* `eval_interval`：评估间隔

### 预测配置

* `visualize`：是否绘制预测图像
* `save_results`：是否保存预测结果

---

## 📜 文件说明

### `data_loader.py`

* `ForceFieldDataset`：加载图像与力场数据
* `fill_missing_force_values`：缺失值填补
* `get_dataloaders`：返回训练/测试 DataLoader
* `preprocess_image`：预测时图像预处理

### `pinn.py`

* `MechanicsPINN`：PINN 主模型
* 使用 MLP 网络输出力场预测
* 包含二阶与四阶差分算子，用于计算 PDE 残差

### `train.py`

* 构建模型与优化器
* 训练循环，计算数据损失 + PDE 损失
* 定期保存 checkpoint

### `evaluate.py`

* 加载模型并在测试集上计算误差指标
* 输出 MSE / RMSE / MAE / 相对误差
* 保存预测与真实力场的可视化图

### `predict.py`

* 输入单张图像进行预测
* 输出 `.npy` 力场矩阵
* 保存可视化结果（输入图、预测图、真实力场对比）

### `utils.py`

* 日志记录
* 目录管理
* 模型保存与加载
* 可视化函数

### `mini_dataset.py`

* 生成小规模演示数据集（图像+力场矩阵）
* 自动引入缺失值，便于测试填补与预测

---

## 📊 示例输出

* **训练日志**：

  * 损失函数收敛曲线
  * 周期性保存 checkpoint
* **评估指标**：

  * `MSE`、`RMSE`、`MAE`、`RelErr%`
* **预测结果**：

  * `.npy` 力场矩阵
  * `.png` 对比图（输入图像 / 预测力场 / 真实力场）

---

