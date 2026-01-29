# STM32 MNIST手写数字识别项目

本项目实现了一个基于PyTorch的轻量级MNIST手写数字识别模型，专为STM32F407微控制器部署而设计。

## 项目结构

```
PYAI/
├── data/                  # 数据目录
│   ├── MNIST/             # MNIST数据集
│   └── preprocessed/      # 预处理后的数据
├── models/                # 模型目录
│   ├── architectures/     # 模型架构定义
│   ├── trained/           # 训练好的模型
│   └── exported/          # 导出的模型（用于STM32部署）
├── src/                   # 源代码目录
│   ├── data_preprocessing/    # 数据预处理模块
│   ├── training/              # 模型训练模块
│   ├── evaluation/            # 模型评估模块
│   ├── deployment/            # 模型部署准备模块
│   └── utils/                 # 工具函数
├── config/                # 配置文件目录
├── scripts/               # 脚本目录
│   ├── train.py           # 训练脚本
│   ├── evaluate.py        # 评估脚本
│   ├── export.py           # 模型导出脚本
│   └── run_workflow.py     # 完整工作流程脚本
├── requirements.txt       # 依赖项文件
├── .env                   # 环境配置文件
└── README.md              # 项目说明文件
```

## 项目功能

1. **数据预处理**：自动下载、加载和预处理MNIST数据集
2. **模型训练**：支持轻量级和极小化两种模型架构，支持GPU加速训练
3. **模型评估**：生成准确率、混淆矩阵和分类报告等评估指标
4. **模型导出**：导出为ONNX格式，支持STM32F407部署
5. **自动化流程**：提供完整的训练、评估和导出自动化脚本

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (可选，用于GPU加速)

## 数据集说明

本项目支持两种方式获取MNIST数据集：

1. **自动下载**：运行训练脚本时，项目会自动检查本地是否存在MNIST数据集。如果不存在，会自动从PyTorch官方源下载。

2. **手动下载**：如果自动下载失败，可以手动运行以下命令下载MNIST数据集：

   ```bash
   # 下载官方MNIST数据集
   python scripts/download_official_mnist.py
   
   # 或者使用备用下载脚本
   python scripts/download_mnist.py
   ```

3. **自定义数据集**：项目提供了自定义手写数字数据集的压缩包 `data/custom/unified_dataset.zip`，使用方法：
   
   ```bash
   # 解压自定义数据集
   Expand-Archive -Path "data/custom/unified_dataset.zip" -DestinationPath "data/custom/" -Force
   ```
   
   解压后，自定义数据集会位于 `data/custom/unified_dataset` 目录下，可以与MNIST数据集混合使用。

## 安装依赖

```bash
# 使用国内镜像源安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 使用方法

### 1. 完整工作流程

运行完整的训练、评估和导出流程：

```bash
python scripts/run_workflow.py --model lightweight --epochs 10 --batch-size 64
```

### 2. 单独运行训练

```bash
python scripts/train.py --model lightweight --epochs 10 --batch-size 64
```

### 3. 单独运行评估

```bash
python scripts/evaluate.py --model lightweight --model-path models/trained/mnist_model_final.pth
```

### 4. 单独运行导出

```bash
python scripts/export.py --model lightweight --model-path models/trained/mnist_model_final.pth
```

## 模型架构

本项目提供两种模型架构：

1. **Lightweight模型**：两层卷积网络，适合对精度有一定要求的场景
2. **Tiny模型**：单层卷积网络，适合资源极其受限的场景

## STM32部署

1. 运行导出脚本生成ONNX模型和部署文件
2. 打开STM32CubeIDE
3. 创建或打开STM32F407项目
4. 打开X-Cube-AI插件
5. 导入`models/exported/stm32/mnist_model.onnx`文件
6. 配置AI模型参数
7. 生成代码并集成到项目中
8. 编译并烧录到STM32F407开发板

## 配置管理

项目使用`.env`文件管理配置参数，可根据需要修改以下配置：

- `DATA_DIR`：数据集目录
- `BATCH_SIZE`：批处理大小
- `EPOCHS`：训练轮数
- `MODEL_ARCHITECTURE`：模型架构（lightweight或tiny）
- `DEVICE`：训练设备（cuda或cpu）

## 性能指标

- **Lightweight模型**：准确率约99%，模型大小约1.5MB
- **Tiny模型**：准确率约97%，模型大小约0.3MB

## 注意事项

1. 确保STM32F407有足够的内存来运行模型
2. 输入数据需要与训练时的预处理方式一致
3. 模型输入尺寸为28x28的灰度图像
4. 建议使用GPU加速训练以提高速度

## 许可证

本项目仅供学习和研究使用，请勿用于商业用途。