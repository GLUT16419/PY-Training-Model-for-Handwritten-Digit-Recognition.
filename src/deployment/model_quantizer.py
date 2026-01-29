import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.quantization import QuantStub, DeQuantStub
import numpy as np

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

class QuantizableSTM32CompatibleMNISTModel(nn.Module):
    """可量化的STM32兼容MNIST模型"""
    
    def __init__(self, input_shape=(1, 28, 28), num_classes=10, use_depthwise=False):
        """
        初始化模型
        Args:
            input_shape: 输入数据形状 (C, H, W)
            num_classes: 类别数量
            use_depthwise: 是否使用深度可分离卷积
        """
        super(QuantizableSTM32CompatibleMNISTModel, self).__init__()
        
        # 量化相关
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # 计算输入特征维度
        self.input_shape = input_shape
        self.use_depthwise = use_depthwise
        
        # 轻量级卷积层，移除批归一化以减少内存使用
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConv(1, 16, kernel_size=3, stride=1, padding=1)  # 减少通道数从20到16
            self.conv2 = DepthwiseSeparableConv(16, 32, kernel_size=3, stride=1, padding=1)  # 减少通道数从40到32
            self.conv3 = DepthwiseSeparableConv(32, 40, kernel_size=3, stride=1, padding=1)  # 进一步减少通道数从48到40
        else:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 减少通道数从20到16
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 减少通道数从40到32
            self.conv3 = nn.Conv2d(32, 40, kernel_size=3, stride=1, padding=1)  # 进一步减少通道数从48到40
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算全连接层输入维度
        # 经过两次池化后，特征图大小变为 28/2/2 = 7
        self.fc1 = nn.Linear(40 * 7 * 7, 80)  # 进一步减少神经元从96到80
        self.dropout = nn.Dropout(p=0.2)  # 添加 Dropout 层
        self.fc2 = nn.Linear(80, num_classes)
        
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
        # 量化输入
        x = self.quant(x)
        
        # 第一层卷积 + 激活 + 池化
        x = self.pool(torch.relu(self.conv1(x)))
        
        # 第二层卷积 + 激活 + 池化
        x = self.pool(torch.relu(self.conv2(x)))
        
        # 第三层卷积 + 激活
        x = torch.relu(self.conv3(x))
        
        # 展平特征图
        x = x.view(-1, 40 * 7 * 7)
        
        # 全连接层 + 激活 + Dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 输出层
        x = self.fc2(x)
        
        # 反量化输出
        x = self.dequant(x)
        
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
    
    def fuse_model(self):
        """融合模型中的操作，提高量化精度"""
        try:
            # 融合卷积和激活操作
            torch.quantization.fuse_modules(self, [['conv1', 'relu'], ['conv2', 'relu'], ['conv3', 'relu'], ['fc1', 'relu']], inplace=True)
        except Exception as e:
            print(f"Error during model fusion: {e}")
    
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

class ModelQuantizer:
    """模型量化器"""
    
    def __init__(self):
        """初始化量化器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def quantize_ptq(self, model, calibration_data, quantized_model_path=None):
        """
        Post-training Quantization (PTQ)
        Args:
            model: 训练好的模型
            calibration_data: 用于校准的数据集
            quantized_model_path: 量化模型保存路径
        Returns:
            quantized_model: 量化后的模型
        """
        try:
            # 创建量化配置
            quantization_config = quantization.QConfig(
                activation=quantization.observer.MinMaxObserver.with_args(dtype=torch.quint8),
                weight=quantization.observer.MinMaxObserver.with_args(dtype=torch.qint8)
            )
            
            # 准备模型
            model.eval()
            
            # 应用量化配置
            model.qconfig = quantization_config
            
            # 准备校准数据
            def calibrate(model, data_loader):
                """使用校准数据进行量化校准"""
                model.eval()
                # 确保模型在正确的设备上
                model.to(self.device)
                with torch.no_grad():
                    for batch in data_loader:
                        try:
                            inputs, _ = batch
                            inputs = inputs.to(self.device)
                            model(inputs)
                        except Exception as e:
                            print(f"Error during calibration: {e}")
                            continue
            
            # 进行校准
            print("开始PTQ校准...")
            calibrate(model, calibration_data)
            
            # 转换为量化模型
            quantized_model = quantization.convert(model, inplace=False)
            
            # 保存量化模型
            if quantized_model_path:
                try:
                    torch.save(quantized_model.state_dict(), quantized_model_path)
                    print(f"量化模型保存到: {quantized_model_path}")
                except Exception as e:
                    print(f"Error saving quantized model: {e}")
            
            return quantized_model
        except Exception as e:
            print(f"Error during PTQ quantization: {e}")
            import traceback
            traceback.print_exc()
            return model
    
    def quantize_qat(self, model, train_loader, val_loader, epochs=10, learning_rate=0.001, quantized_model_path=None):
        """
        Quantization-Aware Training (QAT)
        Args:
            model: 训练好的模型
            train_loader: 训练数据集
            val_loader: 验证数据集
            epochs: 训练轮数
            learning_rate: 学习率
            quantized_model_path: 量化模型保存路径
        Returns:
            quantized_model: 量化后的模型
        """
        try:
            # 创建量化配置
            quantization_config = quantization.QConfig(
                activation=quantization.observer.MinMaxObserver.with_args(dtype=torch.quint8),
                weight=quantization.observer.MinMaxObserver.with_args(dtype=torch.qint8)
            )
            
            # 应用量化配置
            model.qconfig = quantization_config
            
            # 准备量化感知训练
            quantization.prepare_qat(model, inplace=True)
            
            # 定义优化器和损失函数
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # 量化感知训练
            print("开始QAT训练...")
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for batch in train_loader:
                    try:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        
                        running_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                    except Exception as e:
                        print(f"Error during QAT batch processing: {e}")
                        continue
                
                train_accuracy = 100. * correct / total
                print(f"QAT Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.3f}, Accuracy: {train_accuracy:.2f}%")
                
                # 验证
                model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        try:
                            inputs, targets = batch
                            inputs = inputs.to(self.device)
                            targets = targets.to(self.device)
                            outputs = model(inputs)
                            _, predicted = outputs.max(1)
                            val_total += targets.size(0)
                            val_correct += predicted.eq(targets).sum().item()
                        except Exception as e:
                            print(f"Error during QAT validation: {e}")
                            continue
                
                val_accuracy = 100. * val_correct / val_total
                print(f"Validation Accuracy: {val_accuracy:.2f}%")
            
            # 转换为量化模型
            quantized_model = quantization.convert(model.eval(), inplace=False)
            
            # 保存量化模型
            if quantized_model_path:
                try:
                    torch.save(quantized_model.state_dict(), quantized_model_path)
                    print(f"量化模型保存到: {quantized_model_path}")
                except Exception as e:
                    print(f"Error saving quantized model: {e}")
            
            return quantized_model
        except Exception as e:
            print(f"Error during QAT quantization: {e}")
            import traceback
            traceback.print_exc()
            return model
    
    def evaluate_quantized_model(self, model, test_loader):
        """
        评估量化模型性能
        Args:
            model: 量化模型
            test_loader: 测试数据集
        Returns:
            accuracy: 准确率
            avg_loss: 平均损失
        """
        try:
            model.eval()
            correct = 0
            total = 0
            total_loss = 0
            criterion = nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for batch in test_loader:
                    try:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        total_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                    except Exception as e:
                        print(f"Error during evaluation: {e}")
                        continue
            
            accuracy = 100. * correct / total
            avg_loss = total_loss / len(test_loader)
            
            print(f"量化模型评估结果:")
            print(f"准确率: {accuracy:.2f}%")
            print(f"平均损失: {avg_loss:.3f}")
            
            return accuracy, avg_loss
        except Exception as e:
            print(f"Error evaluating quantized model: {e}")
            return 0.0, float('inf')
    
    def compare_models(self, float_model, quantized_model, test_loader):
        """
        比较浮点模型和量化模型的性能
        Args:
            float_model: 浮点模型
            quantized_model: 量化模型
            test_loader: 测试数据集
        Returns:
            float_accuracy: 浮点模型准确率
            quantized_accuracy: 量化模型准确率
            accuracy_diff: 准确率差异
        """
        try:
            # 评估浮点模型
            float_model.eval()
            float_correct = 0
            float_total = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    try:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        outputs = float_model(inputs)
                        _, predicted = outputs.max(1)
                        float_total += targets.size(0)
                        float_correct += predicted.eq(targets).sum().item()
                    except Exception as e:
                        print(f"Error evaluating float model: {e}")
                        continue
            
            float_accuracy = 100. * float_correct / float_total
            
            # 评估量化模型
            quantized_accuracy, _ = self.evaluate_quantized_model(quantized_model, test_loader)
            
            # 计算准确率差异
            accuracy_diff = abs(float_accuracy - quantized_accuracy)
            
            print(f"模型性能比较:")
            print(f"浮点模型准确率: {float_accuracy:.2f}%")
            print(f"量化模型准确率: {quantized_accuracy:.2f}%")
            print(f"准确率差异: {accuracy_diff:.2f}%")
            
            return float_accuracy, quantized_accuracy, accuracy_diff
        except Exception as e:
            print(f"Error comparing models: {e}")
            return 0.0, 0.0, 0.0

def get_quantizable_model(model_name='stm32', input_shape=(1, 28, 28), num_classes=10, use_depthwise=False):
    """
    获取可量化模型实例
    Args:
        model_name: 模型名称
        input_shape: 输入数据形状
        num_classes: 类别数量
        use_depthwise: 是否使用深度可分离卷积
    Returns:
        model: 可量化模型实例
    """
    try:
        if model_name == 'stm32':
            return QuantizableSTM32CompatibleMNISTModel(input_shape, num_classes, use_depthwise)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    except Exception as e:
        print(f"Error creating quantizable model: {e}")
        # 返回默认模型
        return QuantizableSTM32CompatibleMNISTModel(input_shape, num_classes, use_depthwise)

if __name__ == "__main__":
    """测试量化功能"""
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    
    from src.data_preprocessing.data_processor import MNISTDataProcessor
    
    # 加载数据
    processor = MNISTDataProcessor(batch_size=32)
    train_loader, val_loader, test_loader = processor.load_data()
    
    # 创建模型
    model = QuantizableSTM32CompatibleMNISTModel()
    
    # 测试PTQ
    quantizer = ModelQuantizer()
    quantized_model_ptq = quantizer.quantize_ptq(model, val_loader)
    
    # 评估量化模型
    quantizer.evaluate_quantized_model(quantized_model_ptq, test_loader)
    
    # 测试QAT
    quantized_model_qat = quantizer.quantize_qat(model, train_loader, val_loader, epochs=5)
    
    # 评估量化模型
    quantizer.evaluate_quantized_model(quantized_model_qat, test_loader)
    
    # 比较模型性能
    quantizer.compare_models(model, quantized_model_ptq, test_loader)
    quantizer.compare_models(model, quantized_model_qat, test_loader)