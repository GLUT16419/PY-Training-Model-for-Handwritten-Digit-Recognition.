import os
import sys
import matplotlib.pyplot as plt
import torch

# 添加项目根目录到Python搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import MNISTDataProcessor

def test_data_loading():
    """测试数据加载和预处理是否正确"""
    print("Testing data loading...")
    
    # 创建数据处理器
    processor = MNISTDataProcessor()
    
    # 加载数据
    train_loader, val_loader, test_loader = processor.load_data()
    
    # 获取数据形状
    input_shape, num_classes = processor.get_data_shape()
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 检查数据加载器
    print("\nChecking data loader...")
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels[:10]}")
        
        # 检查标签是否在0-9范围内
        label_range = torch.unique(labels)
        print(f"  Unique labels in batch: {label_range}")
        print(f"  All labels in 0-9 range: {torch.all((labels >= 0) & (labels <= 9))}")
        
        # 检查图像值范围
        img_min = images.min().item()
        img_max = images.max().item()
        print(f"  Image value range: [{img_min:.4f}, {img_max:.4f}]")
        
        # 只检查第一个批次
        break
    
    print("\nData loading test completed successfully!")

if __name__ == "__main__":
    test_data_loading()