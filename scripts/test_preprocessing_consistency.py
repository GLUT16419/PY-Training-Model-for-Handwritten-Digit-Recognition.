#!/usr/bin/env python3
"""
数据预处理一致性测试脚本
验证训练和测试时使用相同的数据预处理方法
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.testing.model_loader import ModelLoader
from src.data_preprocessing.data_processor import CustomMNISTDataset

def test_preprocessing_consistency():
    """测试数据预处理一致性"""
    print("=== 数据预处理一致性测试 ===")
    
    # 创建测试图像
    test_image_path = 'test_preprocessing.png'
    
    # 创建一个简单的测试图像（数字5）
    from PIL import Image, ImageDraw, ImageFont
    
    # 创建一个黑色背景的图像
    img = Image.new('L', (100, 100), color=255)
    d = ImageDraw.Draw(img)
    
    # 尝试加载字体，如果失败则使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        font = ImageFont.load_default()
    
    # 绘制数字5
    d.text((20, 10), "5", font=font, fill=0)
    
    # 保存测试图像
    img.save(test_image_path)
    print(f"创建测试图像: {test_image_path}")
    
    # 1. 使用训练时的数据预处理
    print("\n1. 使用训练时的数据预处理:")
    try:
        # 创建自定义数据集实例
        custom_dataset = CustomMNISTDataset(root_dir='data/custom', transform=None)
        
        # 预处理图像
        train_processed = custom_dataset._preprocess_image(img)
        train_processed_np = np.array(train_processed)
        
        # 应用默认转换
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_transformed = transform(train_processed)
        train_transformed_np = train_transformed.numpy()[0]
        
        print(f"训练预处理后形状: {train_transformed_np.shape}")
        print(f"训练预处理后均值: {train_transformed_np.mean():.4f}")
        print(f"训练预处理后标准差: {train_transformed_np.std():.4f}")
        
    except Exception as e:
        print(f"训练预处理测试失败: {str(e)}")
    
    # 2. 使用测试时的数据预处理
    print("\n2. 使用测试时的数据预处理:")
    try:
        # 创建模型加载器实例
        model_loader = ModelLoader()
        
        # 预处理图像
        test_processed = model_loader.preprocess_image(img)
        
        print(f"测试预处理后形状: {test_processed.shape}")
        print(f"测试预处理后均值: {test_processed.mean():.4f}")
        print(f"测试预处理后标准差: {test_processed.std():.4f}")
        
    except Exception as e:
        print(f"测试预处理测试失败: {str(e)}")
    
    # 3. 比较两种预处理方法的结果
    print("\n3. 比较预处理结果:")
    try:
        # 计算差异
        diff = np.abs(train_transformed_np - test_processed)
        mean_diff = diff.mean()
        max_diff = diff.max()
        
        print(f"平均差异: {mean_diff:.4f}")
        print(f"最大差异: {max_diff:.4f}")
        
        if mean_diff < 0.01 and max_diff < 0.1:
            print("✓ 预处理方法一致！")
        else:
            print("✗ 预处理方法不一致！")
            
    except Exception as e:
        print(f"比较失败: {str(e)}")
    
    # 清理测试图像
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
        print(f"\n清理测试图像: {test_image_path}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_preprocessing_consistency()
