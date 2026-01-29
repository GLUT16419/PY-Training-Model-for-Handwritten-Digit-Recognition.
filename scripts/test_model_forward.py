import os
import sys
import torch

# 添加项目根目录到Python搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training import get_model

def test_model_forward():
    """测试模型前向传播是否正常"""
    print("Testing model forward pass...")
    
    # 测试简单模型
    print("\nTesting SimpleMNISTModel...")
    simple_model = get_model('simple', (1, 28, 28), 10)
    print(f"Model created: {simple_model.__class__.__name__}")
    
    # 创建一个随机输入
    input_tensor = torch.randn(1, 1, 28, 28)
    print(f"Input shape: {input_tensor.shape}")
    
    # 前向传播
    output = simple_model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    print(f"Output sum: {output.sum().item()}")
    print(f"Output max: {output.max().item()}")
    print(f"Output min: {output.min().item()}")
    
    # 测试轻量级模型
    print("\nTesting LightweightMNISTModel...")
    lightweight_model = get_model('lightweight', (1, 28, 28), 10)
    print(f"Model created: {lightweight_model.__class__.__name__}")
    
    # 前向传播
    output = lightweight_model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    print(f"Output sum: {output.sum().item()}")
    print(f"Output max: {output.max().item()}")
    print(f"Output min: {output.min().item()}")
    
    print("\nModel forward pass test completed!")

if __name__ == "__main__":
    test_model_forward()