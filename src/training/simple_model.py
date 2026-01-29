import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMNISTModel(nn.Module):
    """简单的MNIST分类模型，使用全连接层"""
    
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        """
        初始化模型
        Args:
            input_shape: 输入数据形状 (C, H, W)
            num_classes: 类别数量
        """
        super(SimpleMNISTModel, self).__init__()
        
        # 计算输入特征维度
        self.input_shape = input_shape
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        
        # 全连接层
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入数据
        Returns:
            output: 模型输出
        """
        # 展平输入
        x = x.view(x.size(0), -1)
        
        # 全连接层 + 激活
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 输出层
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def get_model(model_name='lightweight', input_shape=(1, 28, 28), num_classes=10):
    """
    获取模型实例
    Args:
        model_name: 模型名称 ('lightweight', 'tiny' 或 'simple')
        input_shape: 输入数据形状
        num_classes: 类别数量
    Returns:
        model: 模型实例
    """
    if model_name == 'lightweight':
        from .model_architectures import LightweightMNISTModel
        return LightweightMNISTModel(input_shape, num_classes)
    elif model_name == 'tiny':
        from .model_architectures import TinyMNISTModel
        return TinyMNISTModel(input_shape, num_classes)
    elif model_name == 'simple':
        return SimpleMNISTModel(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")