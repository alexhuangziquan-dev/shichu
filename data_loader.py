import os
import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from scipy.ndimage import zoom
from config import DATA_CONFIG, BASE_CONFIG


def fill_missing_force_values(force_matrix, decay_rate=0.1):
    """

    """
    # 复制原始矩阵，避免修改输入数据（保护已知点）
    filled_matrix = force_matrix.copy()
    h, w = filled_matrix.shape

    # 1. 提取所有有效参考点：非-1（非缺失）且非0（非无效值）的已知点
    valid_reference_points = []
    for i in range(h):
        for j in range(w):
            val = filled_matrix[i, j]
            if val != -1 and val != 0:  # 仅保留非缺失、非0的有效点
                valid_reference_points.append((i, j, val))

    # 无有效参考点时的处理：填充为0（避免后续计算报错，同时提示用户）
    if not valid_reference_points:
        print("警告：当前力场数据无有效参考点（无'非-1且非0'的已知点），所有缺失值填充为0")
        filled_matrix[filled_matrix == -1] = 0
        return filled_matrix.astype(np.float32)

    # 2. 遍历所有缺失点（值为-1的位置）执行填充
    missing_positions = np.argwhere(filled_matrix == -1)  # 获取所有缺失点坐标 (N×2: [行, 列])
    for (i, j) in missing_positions:
        # 计算当前缺失点与所有有效参考点的欧氏距离（图像像素距离）
        min_distance = float('inf')
        nearest_reference_val = 0.0

        for (ref_i, ref_j, ref_val) in valid_reference_points:
            # 欧氏距离公式：distance = sqrt((当前行-参考行)² + (当前列-参考列)²)
            distance = math.hypot(i - ref_i, j - ref_j)

            # 更新最近参考点（距离更小则替换）
            if distance < min_distance:
                min_distance = distance
                nearest_reference_val = ref_val

        # 3. 按最近参考点和距离计算填充值（指数衰减）
        # 距离为0时直接取参考值（理论上不会触发，因参考点非-1）
        if min_distance == 0:
            fill_val = nearest_reference_val
        else:
            # 指数衰减公式：fill_val = 最近参考值 × e^(-衰减率 × 距离)
            # 衰减率越大，值随距离下降越快
            fill_val = nearest_reference_val * np.exp(-decay_rate * min_distance)

        # 赋值到填充矩阵（仅修改缺失点，保护已知点）
        filled_matrix[i, j] = fill_val

    # 确保输出为float32类型（匹配PyTorch模型输入要求）
    return filled_matrix.astype(np.float32)


class ForceFieldDataset(Dataset):
    """力场预测数据集（优化版：严格匹配缺失值填充需求）"""

    def __init__(self, image_dir, force_dir, image_size=(1080, 1920),
                 transform=None, fill_missing=True, decay_rate=0.1):
        self.image_dir = image_dir
        self.force_dir = force_dir
        self.image_size = image_size  # (H, W) 对应1920×1080分辨率
        self.transform = transform
        self.fill_missing = fill_missing  # 是否启用缺失值填充
        self.decay_rate = decay_rate  # 指数衰减率（值越大衰减越快）

        # 加载并排序图像/力场路径（按文件名排序，确保一一对应）
        self.image_paths = self._get_sorted_paths(image_dir, [".png", ".jpg"])
        self.force_paths = self._get_sorted_paths(force_dir, [".npy"])

        # 基础校验：图像与力场数量必须一致
        assert len(self.image_paths) == len(self.force_paths), \
            f"图像数量({len(self.image_paths)})与力场标签数量({len(self.force_paths)})不匹配"

        # 提前校验力场数据格式（避免训练时批量报错）
        self._precheck_force_data()

    def _get_sorted_paths(self, dir_path, extensions):
        """获取指定目录下指定格式的文件路径，按文件名（不含后缀）升序排序"""
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"目录不存在：{dir_path}")

        paths = []
        for ext in extensions:
            # 匹配所有指定后缀的文件
            ext_paths = glob.glob(os.path.join(dir_path, f"*{ext}"))
            paths.extend(ext_paths)

        if not paths:
            raise FileNotFoundError(f"目录 {dir_path} 下无{extensions}格式文件")

        # 按文件名（不含后缀）排序（确保图像与力场一一对应，如img_001.jpg对应img_001.npy）
        paths.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
        return paths

    def _precheck_force_data(self):
        """提前校验力场数据：维度、缺失值标识、是否含有效参考点"""
        # 抽样校验前3个力场文件（覆盖大部分格式问题）
        sample_force_paths = self.force_paths[:3]
        for path in sample_force_paths:
            force_data = np.load(path)

            # 校验1：必须是2D矩阵（H×W，与图像尺寸对应）
            assert force_data.ndim == 2, \
                f"力场文件 {os.path.basename(path)} 格式错误：需2D矩阵（实际{force_data.ndim}D）"

            # 校验2：数据类型必须是数值类型（int/float均可，后续统一转float32）
            assert np.issubdtype(force_data.dtype, np.number), \
                f"力场文件 {os.path.basename(path)} 数据类型错误：需数值类型（实际{force_data.dtype}）"

            # 校验3：缺失值标识必须是-1（若存在缺失）
            unique_vals = np.unique(force_data)
            if -1 in unique_vals:
                # 统计缺失值比例（仅作提前告知，不阻断程序）
                missing_ratio = (force_data == -1).sum() / (force_data.shape[0] * force_data.shape[1])
                print(f"提前校验：力场文件 {os.path.basename(path)} 缺失率{missing_ratio:.1%}（标识为-1）")

            # 校验4：是否含有效参考点（非-1且非0）
            valid_ref_count = ((force_data != -1) & (force_data != 0)).sum()
            if valid_ref_count == 0:
                raise ValueError(f"力场文件 {os.path.basename(path)} 无有效参考点（需含'非-1且非0'的已知点）")

    def __len__(self):
        """数据集样本总数"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """加载单样本：图像预处理 + 力场加载与填充"""
        # -------------------------- 1. 图像加载与预处理 --------------------------
        img_path = self.image_paths[idx]
        try:
            with Image.open(img_path) as img:
                # 步骤1：调整尺寸为目标大小（1080×1920），双线性插值保持图像平滑
                img_resized = img.resize(self.image_size, Image.BILINEAR)
                # 步骤2：转为RGB通道（避免灰度图单通道问题，统一输入格式）
                img_rgb = img_resized.convert("RGB")
                # 步骤3：转为numpy数组并归一化到[0,1]（模型训练更稳定，梯度不易爆炸）
                img_np = np.array(img_rgb, dtype=np.float32) / 255.0
                # 步骤4：转为PyTorch格式（C×H×W，原格式为H×W×C）
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        except Exception as e:
            raise RuntimeError(f"加载图像 {os.path.basename(img_path)} 失败：{str(e)}")

        # -------------------------- 2. 力场加载与缺失值填充 --------------------------
        force_path = self.force_paths[idx]
        try:
            # 加载原始力场数据（保留原始值，不直接修改）
            force_original = np.load(force_path)

            # 统计关键信息（用于日志和校验）
            total_pixels = force_original.shape[0] * force_original.shape[1]
            missing_count = (force_original == -1).sum()
            missing_ratio = missing_count / total_pixels
            valid_ref_count = ((force_original != -1) & (force_original != 0)).sum()

            # 若启用填充且存在缺失值
            if self.fill_missing and missing_count > 0:
                # 日志提示：仅当缺失率>10%时打印（避免过多冗余日志）
                if missing_ratio > 0.1:
                    print(
                        f"处理力场 {os.path.basename(force_path)}：缺失率{missing_ratio:.1%}，有效参考点{valid_ref_count}个")

                # 执行填充（核心函数，不修改原始已知点）
                force_filled = fill_missing_force_values(force_original, self.decay_rate)
            else:
                # 不启用填充时：将缺失值(-1)替换为0，避免模型输入异常
                force_filled = np.where(force_original == -1, 0, force_original).astype(np.float32)

            # 校验并调整力场尺寸（必须与图像尺寸一致）
            if force_filled.shape != self.image_size:
                print(
                    f"调整力场 {os.path.basename(force_path)} 尺寸：{force_filled.shape} → {self.image_size}（双线性插值）")
                force_filled = zoom(
                    force_filled,
                    zoom=(self.image_size[0] / force_filled.shape[0], self.image_size[1] / force_filled.shape[1]),
                    order=1  # 1=双线性插值（平衡精度与速度，适合力场分布）
                )

            # 转为PyTorch张量（匹配模型输入）
            force_tensor = torch.from_numpy(force_filled)
        except Exception as e:
            raise RuntimeError(f"加载力场 {os.path.basename(force_path)} 失败：{str(e)}")

        # -------------------------- 3. 应用图像变换（可选） --------------------------
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # 返回：预处理后的图像张量 + 填充后的力场张量
        return img_tensor, force_tensor


def get_dataloaders(fill_missing=None, decay_rate=None):
    """
    获取训练/测试数据加载器（支持临时调整参数，无需修改配置文件）
    参数优先级：函数传入参数 > 配置文件参数（方便调试）
    """
    # 从配置文件获取默认参数（若函数未传入）
    final_fill_missing = fill_missing if fill_missing is not None else DATA_CONFIG["fill_missing"]
    final_decay_rate = decay_rate if decay_rate is not None else DATA_CONFIG["decay_rate"]

    # 构建数据集路径（按配置文件定义的结构）
    train_image_dir = os.path.join(DATA_CONFIG["data_path"], "train", "images")
    train_force_dir = os.path.join(DATA_CONFIG["data_path"], "train", "forces")
    test_image_dir = os.path.join(DATA_CONFIG["data_path"], "test", "images")
    test_force_dir = os.path.join(DATA_CONFIG["data_path"], "test", "forces")

    # 创建训练集（启用打乱，增强模型泛化能力）
    train_dataset = ForceFieldDataset(
        image_dir=train_image_dir,
        force_dir=train_force_dir,
        image_size=DATA_CONFIG["image_size"],
        fill_missing=final_fill_missing,
        decay_rate=final_decay_rate
    )

    # 创建测试集（禁用打乱，确保评估结果可复现）
    test_dataset = ForceFieldDataset(
        image_dir=test_image_dir,
        force_dir=test_force_dir,
        image_size=DATA_CONFIG["image_size"],
        fill_missing=final_fill_missing,
        decay_rate=final_decay_rate
    )

    # 构建数据加载器（多线程加速，适配GPU）
    train_loader = DataLoader(
        train_dataset,
        batch_size=DATA_CONFIG["batch_size"],
        shuffle=True,
        num_workers=min(4, os.cpu_count()),  # 线程数不超过CPU核心数，避免内存溢出
        pin_memory=True,  # 启用内存锁定，加速GPU数据传输（仅当使用GPU时生效）
        drop_last=True  # 丢弃最后一个不完整批次（避免批次大小不一致导致的训练报错）
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=DATA_CONFIG["batch_size"],
        shuffle=False,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True,
        drop_last=False  # 测试集保留所有样本，确保评估完整
    )

    # 打印数据加载器信息（方便调试，确认配置是否正确）
    print("=" * 50)
    print("数据加载器配置完成：")
    print(f"  训练集：{len(train_dataset)}个样本 → {len(train_loader)}个批次")
    print(f"  测试集：{len(test_dataset)}个样本 → {len(test_loader)}个批次")
    print(f"  缺失值填充：{'启用' if final_fill_missing else '禁用'}")
    print(f"  指数衰减率：{final_decay_rate}")
    print(f"  目标图像尺寸：{DATA_CONFIG['image_size']}（H×W）")
    print("=" * 50)

    return train_loader, test_loader


def preprocess_image(image_path):
    """预处理单张预测图像（与数据集预处理逻辑完全一致，确保输入格式统一）"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"预测图像不存在：{image_path}")

    try:
        with Image.open(image_path) as img:
            # 步骤1：调整尺寸为目标大小（1080×1920）
            img_resized = img.resize(DATA_CONFIG["image_size"], Image.BILINEAR)
            # 步骤2：转为RGB通道
            img_rgb = img_resized.convert("RGB")
            # 步骤3：归一化到[0,1]并转为float32
            img_np = np.array(img_rgb, dtype=np.float32) / 255.0
            # 步骤4：转为PyTorch格式（C×H×W）并增加批次维度（1×C×H×W）
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        # 移动到指定设备（GPU/CPU，与模型一致）
        return img_tensor.to(BASE_CONFIG["device"])
    except Exception as e:
        raise RuntimeError(f"预处理预测图像 {os.path.basename(image_path)} 失败：{str(e)}")