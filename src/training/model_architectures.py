import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightMNISTModel(nn.Module):
    """轻量级MNIST分类模型，适合STM32F407部署"""
    
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        """
        初始化模型
        Args:
            input_shape: 输入数据形状 (C, H, W)
            num_classes: 类别数量
        """
        super(LightweightMNISTModel, self).__init__()
        
        # 计算输入特征维度
        self.input_shape = input_shape
        
        # 轻量级卷积层，增加复杂度
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 批归一化层
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 添加卷积层后的dropout
        self.dropout_conv = nn.Dropout(p=0.3)  # 增加卷积层dropout率
        
        # 计算全连接层输入维度
        # 经过两次池化后，特征图大小变为 28/2/2 = 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.4)  # 添加 Dropout 层
        self.fc2 = nn.Linear(128, num_classes)
        
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
        # 第一层卷积 + 批归一化 + 激活 + 池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 第二层卷积 + 批归一化 + 激活 + 池化
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 添加卷积层后的dropout
        x = self.dropout_conv(x)
        
        # 展平特征图
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层 + 激活 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 输出层
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def load_state_dict(self, state_dict, strict=False):
        """
        加载模型权重，支持兼容不同结构的模型
        Args:
            state_dict: 模型权重
            strict: 是否严格匹配
        """
        try:
            # 尝试直接加载
            super().load_state_dict(state_dict, strict=strict)
            print("Model weights loaded successfully!")
        except Exception as e:
            print(f"Error loading state dict: {e}")
            print("Attempting to load with compatibility adjustments...")
            
            # 调整状态字典以匹配当前模型结构
            adjusted_state_dict = {}
            for key, value in state_dict.items():
                # 处理深度可分离卷积的权重
                if 'depthwise' in key or 'pointwise' in key:
                    # 如果当前模型使用深度可分离卷积，直接使用
                    if self.use_depthwise:
                        adjusted_state_dict[key] = value
                else:
                    # 对于普通卷积层，根据当前模型结构调整
                    if not self.use_depthwise and 'conv' in key:
                        adjusted_state_dict[key] = value
            
            # 加载调整后的状态字典
            try:
                super().load_state_dict(adjusted_state_dict, strict=False)
                print("Model weights loaded with compatibility adjustments!")
            except Exception as e:
                print(f"Error loading adjusted state dict: {e}")
                print("Loading only matching keys...")
                # 只加载匹配的键
                model_dict = self.state_dict()
                filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(filtered_state_dict)
                super().load_state_dict(model_dict, strict=False)
                print("Model weights loaded with filtered keys!")

class EnhancedLightweightMNISTModel(nn.Module):
    """增强型轻量级MNIST分类模型，适合STM32F407部署"""
    
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        """
        初始化模型
        Args:
            input_shape: 输入数据形状 (C, H, W)
            num_classes: 类别数量
        """
        super(EnhancedLightweightMNISTModel, self).__init__()
        
        # 计算输入特征维度
        self.input_shape = input_shape
        
        # 增强型卷积层，增加特征提取能力
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 批归一化层
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # 批归一化层
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)  # 批归一化层
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算全连接层输入维度
        # 经过两次池化后，特征图大小变为 28/2/2 = 7
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.dropout1 = nn.Dropout(p=0.25)  # 添加 Dropout 层
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.2)  # 添加 Dropout 层
        self.fc3 = nn.Linear(256, num_classes)
        
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
        # 第一层卷积 + 批归一化 + 激活 + 池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 第二层卷积 + 批归一化 + 激活 + 池化
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 第三层卷积 + 批归一化 + 激活
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 第四层卷积 + 批归一化 + 激活
        x = F.relu(self.bn4(self.conv4(x)))
        
        # 展平特征图
        x = x.view(-1, 256 * 7 * 7)
        
        # 全连接层 + 激活 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        # 第二个全连接层 + 激活 + Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # 输出层
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class TinyMNISTModel(nn.Module):
    """极小化MNIST分类模型，适合资源极其受限的设备"""
    
    def __init__(self, input_shape=(1, 28, 28), num_classes=10, use_depthwise=True):
        """
        初始化模型
        Args:
            input_shape: 输入数据形状 (C, H, W)
            num_classes: 类别数量
            use_depthwise: 是否使用深度可分离卷积
        """
        super(TinyMNISTModel, self).__init__()
        
        self.use_depthwise = use_depthwise
        
        # 优化型卷积层
        if use_depthwise:
            # 使用深度可分离卷积减少参数量
            self.conv1 = DepthwiseSeparableConv(1, 16, kernel_size=3, stride=1, padding=1)
            self.conv2 = DepthwiseSeparableConv(16, 24, kernel_size=3, stride=1, padding=1)
            self.conv3 = DepthwiseSeparableConv(24, 32, kernel_size=3, stride=1, padding=1)
        else:
            # 常规卷积但减少通道数
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 卷积层dropout
        self.dropout_conv = nn.Dropout(p=0.3)
        
        # 计算全连接层输入维度
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 减少神经元数量
        self.dropout = nn.Dropout(p=0.4)  # 保持 Dropout 率
        self.fc2 = nn.Linear(128, 64)  # 减少神经元数量
        self.dropout2 = nn.Dropout(p=0.3)
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
        # 第一卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv1(x)))
        
        # 第二卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv2(x)))
        
        # 第三卷积 + 激活 + dropout
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        
        # 展平特征图
        x = x.view(-1, 32 * 7 * 7)
        
        # 全连接层 + 激活 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 第二全连接层 + 激活 + Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # 输出层
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'groups') and m.groups > 1:
                    # 深度可分离卷积使用xavier初始化
                    nn.init.xavier_normal_(m.weight)
                else:
                    # 普通卷积使用kaiming初始化
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class AdvancedLightweightMNISTModel(nn.Module):
    """高级轻量级MNIST分类模型，适合STM32F407部署"""
    
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        """
        初始化模型
        Args:
            input_shape: 输入数据形状 (C, H, W)
            num_classes: 类别数量
        """
        super(AdvancedLightweightMNISTModel, self).__init__()
        
        # 计算输入特征维度
        self.input_shape = input_shape
        
        # 第一阶段：浅层特征提取
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, groups=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),  # 深度可分离卷积
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第二阶段：残差学习块
        self.residual_block = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, groups=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),  # 深度可分离卷积
            nn.BatchNorm2d(64)
        )
        
        # 残差连接的1x1卷积
        self.residual_shortcut = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        
        # 第三阶段：特征增强
        self.stage3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),  # 深度可分离卷积
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 128, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 混合池化层
        self.mixed_pooling = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(128, num_classes)
        
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
        # 第一阶段
        x = self.stage1(x)
        
        # 残差连接
        residual = self.residual_shortcut(x)
        x = self.residual_block(x)
        x = x + residual
        
        # 第三阶段
        x = self.stage3(x)
        
        # 注意力机制
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # 展平特征图
        x = x.view(-1, 128 * 7 * 7)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups == 1:
                    # 普通卷积使用kaiming初始化
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    # 深度可分离卷积使用xavier初始化
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        初始化深度可分离卷积
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
        """
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        # 逐点卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        
    def forward(self, x):
        """前向传播"""
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class STM32CompatibleMNISTModel(nn.Module):
    """STM32兼容的MNIST分类模型，适合STM32F407部署"""
    
    def __init__(self, input_shape=(1, 28, 28), num_classes=10, use_depthwise=False):
        """
        初始化模型
        Args:
            input_shape: 输入数据形状 (C, H, W)
            num_classes: 类别数量
            use_depthwise: 是否使用深度可分离卷积
        """
        super(STM32CompatibleMNISTModel, self).__init__()
        
        # 计算输入特征维度
        self.input_shape = input_shape
        self.use_depthwise = use_depthwise
        
        # 轻量级卷积层，移除批归一化以减少内存使用
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConv(1, 8, kernel_size=3, stride=1, padding=1)  # 大幅减少通道数到8
            self.conv2 = DepthwiseSeparableConv(8, 16, kernel_size=3, stride=1, padding=1)  # 大幅减少通道数到16
            self.conv3 = DepthwiseSeparableConv(16, 20, kernel_size=3, stride=1, padding=1)  # 进一步减少通道数到20
        else:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # 大幅减少通道数到8
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)  # 大幅减少通道数到16
            self.conv3 = nn.Conv2d(16, 20, kernel_size=3, stride=1, padding=1)  # 进一步减少通道数到20
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 添加卷积层后的dropout
        self.dropout_conv = nn.Dropout(p=0.4)  # 增加卷积层dropout率
        
        # 计算全连接层输入维度
        # 经过两次池化后，特征图大小变为 28/2/2 = 7
        self.fc1 = nn.Linear(20 * 7 * 7, 32)  # 进一步减少神经元数量
        self.dropout = nn.Dropout(p=0.6)  # 大幅增加全连接层dropout率
        self.fc2 = nn.Linear(32, num_classes)
        
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
        # 第一层卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv1(x)))
        
        # 第二层卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv2(x)))
        
        # 第三层卷积 + 激活 + dropout
        x = F.relu(self.conv3(x))
        x = self.dropout_conv(x)
        
        # 展平特征图
        x = x.view(-1, 20 * 7 * 7)
        
        # 全连接层 + 激活 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 输出层
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_state_dict(self, state_dict, strict=False):
        """
        加载模型权重，支持兼容不同结构的模型
        Args:
            state_dict: 模型权重
            strict: 是否严格匹配
        """
        try:
            # 尝试直接加载
            super().load_state_dict(state_dict, strict=strict)
            print("Model weights loaded successfully!")
        except Exception as e:
            print(f"Error loading state dict: {e}")
            print("Attempting to load with compatibility adjustments...")
            
            # 调整状态字典以匹配当前模型结构
            adjusted_state_dict = {}
            for key, value in state_dict.items():
                # 处理深度可分离卷积的权重
                if 'depthwise' in key or 'pointwise' in key:
                    # 如果当前模型使用深度可分离卷积，直接使用
                    if self.use_depthwise:
                        adjusted_state_dict[key] = value
                else:
                    # 对于普通卷积层，根据当前模型结构调整
                    if not self.use_depthwise and 'conv' in key:
                        adjusted_state_dict[key] = value
            
            # 加载调整后的状态字典
            try:
                super().load_state_dict(adjusted_state_dict, strict=False)
                print("Model weights loaded with compatibility adjustments!")
            except Exception as e:
                print(f"Error loading adjusted state dict: {e}")
                print("Loading only matching keys...")
                # 只加载匹配的键
                model_dict = self.state_dict()
                filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(filtered_state_dict)
                super().load_state_dict(model_dict, strict=False)
                print("Model weights loaded with filtered keys!")

class UltraLightMNISTModel(nn.Module):
    """超轻量MNIST分类模型，专为STM32F407部署设计（900K以内）"""
    
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        """
        初始化模型
        Args:
            input_shape: 输入数据形状 (C, H, W)
            num_classes: 类别数量
        """
        super(UltraLightMNISTModel, self).__init__()
        
        # 计算输入特征维度
        self.input_shape = input_shape
        
        # 超轻量级卷积层，最小化参数数量
        # 通道数：1→8→16→32（大幅减少参数）
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算全连接层输入维度
        # 经过两次池化后，特征图大小变为 28/2/2 = 7
        self.fc1 = nn.Linear(32 * 7 * 7, 64)  # 减少神经元数量
        self.fc2 = nn.Linear(64, num_classes)
        
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
        # 第一层卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv1(x)))
        
        # 第二层卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv2(x)))
        
        # 第三层卷积 + 激活
        x = F.relu(self.conv3(x))
        
        # 展平特征图
        x = x.view(-1, 32 * 7 * 7)
        
        # 全连接层 + 激活（移除Dropout以减少内存使用）
        x = F.relu(self.fc1(x))
        
        # 输出层
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def get_model(model_name='lightweight', input_shape=(1, 28, 28), num_classes=10, use_depthwise=False):
    """
    获取模型实例
    Args:
        model_name: 模型名称 ('lightweight', 'enhanced', 'tiny', 'advanced', 'stm32', or 'ultralight')
        input_shape: 输入数据形状
        num_classes: 类别数量
        use_depthwise: 是否使用深度可分离卷积（适用于tiny和stm32模型）
    Returns:
        model: 模型实例
    """
    if model_name == 'lightweight':
        return LightweightMNISTModel(input_shape, num_classes)
    elif model_name == 'enhanced':
        return EnhancedLightweightMNISTModel(input_shape, num_classes)
    elif model_name == 'tiny':
        return TinyMNISTModel(input_shape, num_classes, use_depthwise)
    elif model_name == 'advanced':
        return AdvancedLightweightMNISTModel(input_shape, num_classes)
    elif model_name == 'stm32':
        return STM32CompatibleMNISTModel(input_shape, num_classes, use_depthwise)
    elif model_name == 'ultralight':
        return UltraLightMNISTModel(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")