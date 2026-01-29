import torch
from torchvision import transforms

class DataAugmenter:
    """数据增强器"""
    
    def __init__(self):
        """初始化数据增强器"""
        # 定义训练数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 定义验证和测试数据转换（无增强）
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def get_train_transform(self):
        """
        获取训练数据增强转换
        Returns:
            transform: 训练数据转换
        """
        return self.train_transform
    
    def get_test_transform(self):
        """
        获取测试数据转换
        Returns:
            transform: 测试数据转换
        """
        return self.test_transform