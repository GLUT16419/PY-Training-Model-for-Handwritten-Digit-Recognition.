#!/usr/bin/env python3
"""
STM32兼容模型验证脚本
专门测试新训练的STM32CompatibleMNISTModel模型
"""

import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.testing.model_loader import ModelLoader
from src.training.model_architectures import STM32CompatibleMNISTModel

def test_stm32_model():
    """测试STM32兼容模型"""
    print("=== STM32兼容模型验证 ===")
    
    # 创建模型加载器
    model_loader = ModelLoader()
    
    # 找到最新训练的STM32模型
    stm32_model_dir = None
    latest_time = 0
    
    for folder in os.listdir('models/trained'):
        if folder.startswith('train_'):
            folder_path = os.path.join('models/trained', folder)
            if os.path.isdir(folder_path):
                # 检查文件夹中是否有STM32模型
                for file in os.listdir(folder_path):
                    if file.startswith('best_model_') and file.endswith('.pth'):
                        # 从文件名中提取时间戳
                        try:
                            timestamp = folder.split('_')[1] + folder.split('_')[2]
                            timestamp_int = int(timestamp)
                            if timestamp_int > latest_time:
                                latest_time = timestamp_int
                                stm32_model_dir = folder_path
                        except:
                            pass
    
    if not stm32_model_dir:
        print("没有找到STM32模型，请先训练模型")
        return
    
    print(f"找到最新的STM32模型目录: {stm32_model_dir}")
    
    # 找到该目录中准确率最高的模型
    best_model_path = None
    highest_accuracy = 0.0
    
    for file in os.listdir(stm32_model_dir):
        if file.startswith('best_model_') and file.endswith('.pth'):
            try:
                accuracy_str = file.split('_')[2]
                accuracy = float(accuracy_str)
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    best_model_path = os.path.join(stm32_model_dir, file)
            except:
                pass
    
    if not best_model_path:
        print("没有找到最佳模型文件")
        return
    
    print(f"加载最佳STM32模型: {best_model_path}")
    print(f"模型准确率: {highest_accuracy:.2f}%")
    
    # 加载STM32模型
    success = model_loader.load_pytorch_model(best_model_path, 'stm32')
    
    if not success:
        print("无法加载STM32模型")
        return
    
    print("成功加载STM32模型")
    
    # 测试MNIST测试集
    test_mnist(model_loader)
    
    # 测试自定义数据集
    test_custom_dataset(model_loader)
    
    # 测试STM32格式数据
    test_stm32_format(model_loader)

def test_mnist(model_loader):
    """在MNIST测试集上测试模型"""
    print("\n=== 在MNIST测试集上评估STM32模型 ===")
    
    # 加载MNIST测试集
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='data/MNIST',
        train=False,
        download=False,
        transform=test_transform
    )
    
    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )
    
    # 评估模型
    correct = 0
    total = 0
    
    # 直接使用模型的forward方法进行批量评估
    if model_loader.model_type == 'pytorch' and model_loader.model is not None:
        model = model_loader.model
        model.eval()
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(model_loader.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                # 计算准确率
                batch_correct = (predicted == labels.to(model_loader.device)).sum().item()
                correct += batch_correct
                total += labels.size(0)
    
    accuracy = correct / total * 100
    print(f"MNIST测试集准确率: {accuracy:.2f}%")
    print(f"正确预测: {correct}/{total}")

def test_custom_dataset(model_loader):
    """在自定义数据集上测试模型"""
    print("\n=== 在自定义手绘数字上评估STM32模型 ===")
    
    # 自定义数据集路径
    dataset_path = 'data/custom/unified_dataset'
    
    if not os.path.exists(dataset_path):
        print(f"自定义数据集目录不存在: {dataset_path}")
        return
    
    # 评估每个数字
    digit_accuracies = {}
    total_correct = 0
    total_samples = 0
    
    for digit in range(10):
        digit_dir = os.path.join(dataset_path, str(digit))
        
        if not os.path.exists(digit_dir):
            continue
        
        # 获取该数字的所有图像
        image_files = [f for f in os.listdir(digit_dir) if f.endswith('.png')]
        
        if not image_files:
            continue
        
        correct = 0
        
        for image_file in image_files:
            image_path = os.path.join(digit_dir, image_file)
            
            try:
                # 加载图像
                img = Image.open(image_path)
                
                # 使用模型进行预测
                predicted, confidence, _ = model_loader.predict(img)
                
                # 计算准确率
                if predicted == digit:
                    correct += 1
                
            except Exception as e:
                print(f"评估图像 {image_file} 时出错: {str(e)}")
                continue
        
        # 计算该数字的准确率
        digit_accuracy = correct / len(image_files) * 100
        digit_accuracies[digit] = {
            'accuracy': digit_accuracy,
            'correct': correct,
            'total': len(image_files)
        }
        
        print(f"数字 {digit}: {digit_accuracy:.2f}% ({correct}/{len(image_files)})")
        
        total_correct += correct
        total_samples += len(image_files)
    
    # 计算总体准确率
    if total_samples > 0:
        overall_accuracy = total_correct / total_samples * 100
        print(f"\n自定义数据集总体准确率: {overall_accuracy:.2f}% ({total_correct}/{total_samples})")
    else:
        overall_accuracy = 0.0
        print("\n自定义数据集为空，无法计算准确率")
    
    return overall_accuracy, digit_accuracies

def test_stm32_format(model_loader):
    """测试STM32格式的数据处理"""
    print("\n=== 测试STM32格式数据处理 ===")
    
    # 创建测试数据（模拟STM32发送的数据）
    # 创建一个简单的数字5
    test_data = np.zeros(784, dtype=np.uint8)
    
    # 在中心区域绘制数字5
    for i in range(10, 18):
        test_data[28*10 + i] = 255  # 顶部横线
        test_data[28*14 + i] = 255  # 中间横线
        test_data[28*i + 10] = 255  # 左侧竖线
        test_data[28*i + 17] = 255  # 右侧竖线
    
    # 使用STM32专用预测方法
    try:
        predicted, confidence, all_probabilities = model_loader.stm32_predict(test_data)
        print(f"STM32格式数据预测结果:")
        print(f"预测数字: {predicted}")
        print(f"置信度: {confidence:.2%}")
        print(f"各类别概率:")
        for i, prob in enumerate(all_probabilities):
            print(f"数字 {i}: {prob:.2%}")
    except Exception as e:
        print(f"测试STM32格式数据失败: {str(e)}")

if __name__ == "__main__":
    test_stm32_model()
