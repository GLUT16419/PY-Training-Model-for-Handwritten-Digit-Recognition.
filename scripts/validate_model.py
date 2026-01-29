#!/usr/bin/env python3
"""
模型性能验证和诊断脚本
用于评估模型在MNIST测试集和自定义手绘数字上的性能
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
from src.data_preprocessing.data_processor import MNISTDataProcessor

def evaluate_mnist_test_set(model_loader):
    """
    在MNIST测试集上评估模型性能
    Args:
        model_loader: 模型加载器实例
    Returns:
        accuracy: MNIST测试集准确率
        confusion_matrix: 混淆矩阵
    """
    print("=== 在MNIST测试集上评估模型 ===")
    
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
    confusion_matrix = np.zeros((10, 10), dtype=int)
    
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
                
                # 更新混淆矩阵
                for i in range(len(labels)):
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()
                    confusion_matrix[true_label][pred_label] += 1
    else:
        # 如果不是PyTorch模型，使用逐张图像评估
        with torch.no_grad():
            for images, labels in test_loader:
                for i in range(len(images)):
                    # 转换为PIL图像
                    img = transforms.ToPILImage()(images[i])
                    
                    # 使用模型进行预测
                    predicted, confidence, _ = model_loader.predict(img)
                    
                    # 计算准确率
                    if predicted == labels[i].item():
                        correct += 1
                    total += 1
                    
                    # 更新混淆矩阵
                    confusion_matrix[labels[i].item()][predicted] += 1
    
    accuracy = correct / total * 100
    print(f"MNIST测试集准确率: {accuracy:.2f}%")
    print(f"正确预测: {correct}/{total}")
    
    # 打印混淆矩阵
    print("\n混淆矩阵:")
    print("真实值\\预测值 ", end="")
    for i in range(10):
        print(f"{i:4}", end="")
    print()
    
    for i in range(10):
        print(f"{i:10}      ", end="")
        for j in range(10):
            print(f"{confusion_matrix[i][j]:4}", end="")
        print()
    
    return accuracy, confusion_matrix

def evaluate_custom_digits(model_loader):
    """
    在自定义手绘数字上评估模型性能
    Args:
        model_loader: 模型加载器实例
    Returns:
        accuracy: 自定义数据集准确率
        digit_accuracies: 每个数字的准确率
    """
    print("\n=== 在自定义手绘数字上评估模型 ===")
    
    # 检查自定义数据集目录
    custom_data_dir = 'data/custom/unified_dataset'
    
    if not os.path.exists(custom_data_dir):
        print("自定义数据集目录不存在，跳过此评估")
        return 0.0, {}
    
    # 评估每个数字
    digit_accuracies = {}
    total_correct = 0
    total_samples = 0
    
    for digit in range(10):
        digit_dir = os.path.join(custom_data_dir, str(digit))
        
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

def analyze_model_performance(model_loader):
    """
    分析模型性能并生成诊断报告
    Args:
        model_loader: 模型加载器实例
    """
    print("=== 模型性能分析报告 ===")
    
    # 获取模型信息
    model_info = model_loader.get_model_info()
    print(f"模型类型: {model_info.get('type', '未知')}")
    print(f"模型架构: {model_info.get('model', '未知')}")
    print(f"模型路径: {model_info.get('path', '未知')}")
    print(f"设备: {model_info.get('device', '未知')}")
    
    # 评估MNIST测试集
    mnist_accuracy, confusion_matrix = evaluate_mnist_test_set(model_loader)
    
    # 评估自定义数据集
    custom_accuracy, digit_accuracies = evaluate_custom_digits(model_loader)
    
    # 生成诊断报告
    print("\n=== 模型诊断报告 ===")
    print(f"1. MNIST测试集性能: {mnist_accuracy:.2f}%")
    print(f"2. 自定义数据集性能: {custom_accuracy:.2f}%")
    
    # 性能分析
    if mnist_accuracy > 95:
        print("✓ MNIST测试集性能优秀")
    elif mnist_accuracy > 90:
        print("✓ MNIST测试集性能良好")
    else:
        print("✗ MNIST测试集性能需要改进")
    
    if custom_accuracy > 80:
        print("✓ 自定义数据集性能优秀")
    elif custom_accuracy > 60:
        print("✓ 自定义数据集性能良好")
    else:
        print("✗ 自定义数据集性能需要改进")
    
    # 检查模型架构
    model_architecture = model_info.get('model', '')
    if 'Advanced' in model_architecture:
        print("✓ 使用了高级模型架构，具有更好的特征提取能力")
    elif 'Enhanced' in model_architecture:
        print("✓ 使用了增强型模型架构")
    elif 'Lightweight' in model_architecture:
        print("⚠ 使用了轻量级模型架构，可能在复杂数据上表现有限")
    else:
        print("⚠ 使用了基础模型架构，建议考虑更高级的架构")
    
    # 保存诊断报告
    report_path = 'model_diagnostic_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 模型性能诊断报告 ===\n")
        f.write(f"生成时间: {os.popen('date /t').read().strip()}\n")
        f.write(f"模型类型: {model_info.get('type', '未知')}\n")
        f.write(f"模型架构: {model_info.get('model', '未知')}\n")
        f.write(f"模型路径: {model_info.get('path', '未知')}\n")
        f.write(f"设备: {model_info.get('device', '未知')}\n\n")
        
        f.write("=== 性能评估结果 ===\n")
        f.write(f"MNIST测试集准确率: {mnist_accuracy:.2f}%\n")
        f.write(f"自定义数据集准确率: {custom_accuracy:.2f}%\n\n")
        
        f.write("=== 混淆矩阵 ===\n")
        f.write("真实值\\预测值 ")
        for i in range(10):
            f.write(f"{i:4}")
        f.write("\n")
        
        for i in range(10):
            f.write(f"{i:10}      ")
            for j in range(10):
                f.write(f"{confusion_matrix[i][j]:4}")
            f.write("\n")
    
    print(f"\n诊断报告已保存到: {report_path}")
    
    return {
        'mnist_accuracy': mnist_accuracy,
        'custom_accuracy': custom_accuracy,
        'digit_accuracies': digit_accuracies,
        'model_info': model_info
    }

def main():
    """主函数"""
    print("=== 模型性能验证和诊断工具 ===")
    
    # 创建模型加载器
    model_loader = ModelLoader()
    
    # 加载最新模型
    print("加载最新训练的模型...")
    success, model_path = model_loader.load_latest_model('models/trained')
    
    if not success:
        print("无法加载模型，请检查模型文件是否存在")
        return
    
    print(f"成功加载模型: {model_path}")
    
    # 分析模型性能
    results = analyze_model_performance(model_loader)
    
    # 总结结果
    print("\n=== 验证结果总结 ===")
    print(f"MNIST测试集准确率: {results['mnist_accuracy']:.2f}%")
    print(f"自定义数据集准确率: {results['custom_accuracy']:.2f}%")
    
    # 性能建议
    print("\n=== 性能改进建议 ===")
    if results['mnist_accuracy'] < 90:
        print("1. 建议使用更高级的模型架构（如AdvancedLightweightMNISTModel）")
        print("2. 增加训练轮数或调整学习率")
    else:
        print("1. MNIST测试集性能良好，继续保持")
    
    if results['custom_accuracy'] < 60:
        print("3. 建议增加自定义数据的训练比例")
        print("4. 增强数据预处理，提高自定义数据的质量")
        print("5. 使用两阶段训练策略：先在MNIST上预训练，再在自定义数据上微调")
    else:
        print("3. 自定义数据集性能良好，继续优化")
    
    print("\n=== 验证完成 ===")

if __name__ == "__main__":
    main()
