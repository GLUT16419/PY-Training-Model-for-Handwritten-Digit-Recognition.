import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from src.training.model_architectures import get_model

# 尝试导入ONNX Runtime，如果失败则禁用ONNX支持
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("ONNX Runtime 导入成功，支持ONNX模型")
except ImportError as e:
    ONNX_AVAILABLE = False
    print(f"ONNX Runtime 导入失败，禁用ONNX支持: {str(e)}")

class SimpleFCModel(nn.Module):
    """简单全连接模型，用于加载旧的全连接模型检查点"""
    
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        """
        初始化模型
        Args:
            input_shape: 输入数据形状 (C, H, W)
            num_classes: 类别数量
        """
        super(SimpleFCModel, self).__init__()
        
        # 计算输入特征维度
        self.input_shape = input_shape
        input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        
        # 全连接层
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
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
        x = self.fc3(x)
        
        return x

class ModelLoader:
    """模型加载和推理类"""
    
    def __init__(self):
        """初始化模型加载器"""
        self.model = None
        self.session = None
        self.model_type = None  # 'pytorch' or 'onnx'
        self.model_path = None  # 存储模型路径
        self.model_name = None  # 存储模型架构名称
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_error = None  # 存储加载错误信息
    
    def load_pytorch_model(self, model_path, model_name='lightweight'):
        """
        加载PyTorch模型
        Args:
            model_path: 模型文件路径 (.pth)
            model_name: 模型名称 ('lightweight', 'tiny', 'enhanced', 'advanced', 'stm32', or 'simple_fc')
        """
        self.load_error = None
        try:
            if model_name == 'simple_fc':
                # 初始化简单全连接模型
                self.model = SimpleFCModel()
                print(f"初始化简单全连接模型")
            elif model_name == 'tiny':
                # 初始化tiny模型，默认使用深度可分离卷积
                self.model = get_model(model_name, use_depthwise=True)
                print(f"初始化 {model_name} 模型 (use_depthwise=True)")
            elif model_name == 'stm32':
                # 初始化stm32模型，尝试使用深度可分离卷积
                self.model = get_model(model_name, use_depthwise=True)
                print(f"初始化 {model_name} 模型 (use_depthwise=True)")
            else:
                # 初始化标准模型
                self.model = get_model(model_name)
                print(f"初始化 {model_name} 模型")
            
            # 尝试加载模型权重
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            # 设置为评估模式
            self.model.eval()
            self.model.to(self.device)
            self.model_type = 'pytorch'
            self.model_path = model_path
            self.model_name = model_name
            print(f"PyTorch model loaded from {model_path}")
            print(f"模型架构: {self.model.__class__.__name__}")
            print(f"设备: {self.device}")
            return True
        except RuntimeError as e:
            # 记录错误信息
            error_msg = f"加载 {model_name} 模型失败: {str(e)}"
            self.load_error = error_msg
            print(error_msg)
            
            # 如果是tiny模型，尝试不使用深度可分离卷积
            if model_name == 'tiny':
                try:
                    print("尝试使用 use_depthwise=False 加载 tiny 模型...")
                    self.model = get_model(model_name, use_depthwise=False)
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()
                    self.model.to(self.device)
                    self.model_type = 'pytorch'
                    self.model_path = model_path
                    self.model_name = model_name
                    print(f"成功加载 tiny 模型 (use_depthwise=False)")
                    return True
                except Exception as e2:
                    print(f"使用 use_depthwise=False 加载失败: {str(e2)}")
                    pass
            
            # 如果是stm32模型，尝试不使用深度可分离卷积
            if model_name == 'stm32':
                try:
                    print("尝试使用 use_depthwise=False 加载 stm32 模型...")
                    self.model = get_model(model_name, use_depthwise=False)
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()
                    self.model.to(self.device)
                    self.model_type = 'pytorch'
                    self.model_path = model_path
                    self.model_name = model_name
                    print(f"成功加载 stm32 模型 (use_depthwise=False)")
                    return True
                except Exception as e2:
                    print(f"使用 use_depthwise=False 加载失败: {str(e2)}")
                    pass
            
            # 抛出异常，让调用者处理
            raise
    
    def load_onnx_model(self, model_path):
        """
        加载ONNX模型
        Args:
            model_path: 模型文件路径 (.onnx)
        """
        self.load_error = None
        if not ONNX_AVAILABLE:
            error_msg = "ONNX Runtime 不可用，无法加载ONNX模型"
            self.load_error = error_msg
            raise ImportError(error_msg)
        
        try:
            # 创建ONNX运行时会话
            self.session = ort.InferenceSession(model_path)
            self.model_type = 'onnx'
            self.model_path = model_path
            self.model_name = 'onnx'
            print(f"ONNX model loaded from {model_path}")
            print(f"输入形状: {self.session.get_inputs()[0].shape}")
            print(f"输出形状: {self.session.get_outputs()[0].shape}")
            return True
        except Exception as e:
            error_msg = f"加载ONNX模型失败: {str(e)}"
            self.load_error = error_msg
            print(error_msg)
            raise
    
    def preprocess_image(self, image):
        """
        预处理输入图像
        Args:
            image: PIL图像对象
        Returns:
            processed_image: 处理后的图像，适合模型输入
        """
        # 转换为灰度图像
        if image.mode != 'L':
            image = image.convert('L')
        
        # 调整大小为28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 转换为numpy数组
        image_array = np.array(image, dtype=np.float32)
        
        # 添加自适应阈值处理，提高手绘数字的清晰度
        # 使用与训练时相同的自适应阈值处理方法
        image_array = self._adaptive_threshold(image_array)
        
        # 反转颜色（如果需要），确保黑色背景白色数字
        if np.mean(image_array) > 128:
            image_array = 255 - image_array
        
        # 归一化到0-255范围
        if np.max(image_array) > 0:
            image_array = (image_array / np.max(image_array)) * 255
        
        # 转换回PIL图像
        img = Image.fromarray(image_array.astype(np.uint8))
        
        # 应用与训练时相同的转换
        from torchvision import transforms
        # 注意：测试时使用与训练时相同的转换，但不使用随机增强
        # 确保测试时的预处理与训练时保持一致
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        processed_tensor = transform(img)
        
        # 转换为numpy数组
        processed_image = processed_tensor.numpy()[0]
        
        return processed_image
    
    def stm32_preprocess(self, byte_array):
        """
        STM32专用预处理方法，与STM32的处理方式一致
        Args:
            byte_array: 8位字节数组，长度为784 (28x28)
        Returns:
            processed_image: 处理后的图像，适合模型输入
        """
        # 确保输入长度正确
        if len(byte_array) != 784:
            raise ValueError(f"STM32输入数据长度错误: {len(byte_array)}，期望784")
        
        # 直接转换为float数组，与STM32处理方式一致
        image_array = np.array(byte_array, dtype=np.float32)
        
        # 重塑为28x28
        image_array = image_array.reshape(28, 28)
        
        # 应用与训练时相同的归一化
        mean = 0.1307
        std = 0.3081
        image_array = (image_array / 255.0 - mean) / std
        
        return image_array
    
    def _adaptive_threshold(self, img_array, block_size=11, C=10):
        """自适应阈值处理，与训练时使用相同的方法"""
        import numpy as np
        from scipy.ndimage import uniform_filter
        
        # 计算局部均值
        local_mean = uniform_filter(img_array, size=block_size)
        
        # 应用自适应阈值
        thresholded = np.where(img_array > local_mean - C, 255, 0)
        
        return thresholded
    
    def predict(self, image):
        """
        使用模型进行预测
        Args:
            image: PIL图像对象
        Returns:
            predicted_class: 预测的数字类别
            confidence: 预测置信度
            all_probabilities: 所有类别的概率
        """
        if not self.model and not self.session:
            raise ValueError("No model loaded. Please load a model first.")
        
        if self.load_error:
            raise ValueError(f"Model load error: {self.load_error}")
        
        # 预处理图像
        processed_image = self.preprocess_image(image)
        
        if self.model_type == 'pytorch':
            # 转换为PyTorch张量
            input_tensor = torch.tensor(processed_image, dtype=torch.float32)
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
            input_tensor = input_tensor.to(self.device)
            
            # 模型推理
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)
            
            # 转换为numpy数组
            all_probabilities = probabilities.cpu().numpy()[0]
            predicted_class = predicted_class.cpu().numpy()[0]
            confidence = confidence.cpu().numpy()[0]
        
        elif self.model_type == 'onnx' and ONNX_AVAILABLE:
            # 转换为ONNX输入格式
            input_array = processed_image[np.newaxis, np.newaxis, ...]  # (1, 1, 28, 28)
            
            # 获取输入和输出名称
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            # 模型推理
            output = self.session.run([output_name], {input_name: input_array})[0]
            probabilities = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
            confidence = np.max(probabilities, axis=1)[0]
            predicted_class = np.argmax(probabilities, axis=1)[0]
            all_probabilities = probabilities[0]
        
        else:
            raise ValueError("No model loaded or model type not supported. Please load a model first.")
        
        return predicted_class, confidence, all_probabilities
    
    def predict_tensor(self, tensor):
        """
        使用模型直接预测PyTorch张量
        Args:
            tensor: 已经标准化的PyTorch张量 (1, 28, 28)
        Returns:
            predicted_class: 预测的数字类别
            confidence: 预测置信度
            all_probabilities: 所有类别的概率
        """
        if not self.model and not self.session:
            raise ValueError("No model loaded. Please load a model first.")
        
        if self.load_error:
            raise ValueError(f"Model load error: {self.load_error}")
        
        if self.model_type == 'pytorch':
            # 确保张量维度正确
            if len(tensor.shape) == 3:
                tensor = tensor.unsqueeze(0)  # 添加批次维度
            tensor = tensor.to(self.device)
            
            # 模型推理
            with torch.no_grad():
                output = self.model(tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)
            
            # 转换为numpy数组
            all_probabilities = probabilities.cpu().numpy()[0]
            predicted_class = predicted_class.cpu().numpy()[0]
            confidence = confidence.cpu().numpy()[0]
        
        else:
            raise ValueError("This method only supports PyTorch models.")
        
        return predicted_class, confidence, all_probabilities
    
    def stm32_predict(self, byte_array):
        """
        STM32专用预测方法，直接处理STM32格式的数据
        Args:
            byte_array: 8位字节数组，长度为784 (28x28)
        Returns:
            predicted_class: 预测的数字类别
            confidence: 预测置信度
            all_probabilities: 所有类别的概率
        """
        if not self.model and not self.session:
            raise ValueError("No model loaded. Please load a model first.")
        
        if self.load_error:
            raise ValueError(f"Model load error: {self.load_error}")
        
        # 使用STM32专用预处理
        processed_image = self.stm32_preprocess(byte_array)
        
        if self.model_type == 'pytorch':
            # 转换为PyTorch张量
            input_tensor = torch.tensor(processed_image, dtype=torch.float32)
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
            input_tensor = input_tensor.to(self.device)
            
            # 模型推理
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)
            
            # 转换为numpy数组
            all_probabilities = probabilities.cpu().numpy()[0]
            predicted_class = predicted_class.cpu().numpy()[0]
            confidence = confidence.cpu().numpy()[0]
        
        elif self.model_type == 'onnx' and ONNX_AVAILABLE:
            # 转换为ONNX输入格式
            input_array = processed_image[np.newaxis, np.newaxis, ...]  # (1, 1, 28, 28)
            
            # 获取输入和输出名称
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            # 模型推理
            output = self.session.run([output_name], {input_name: input_array})[0]
            probabilities = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
            confidence = np.max(probabilities, axis=1)[0]
            predicted_class = np.argmax(probabilities, axis=1)[0]
            all_probabilities = probabilities[0]
        
        else:
            raise ValueError("No model loaded or model type not supported. Please load a model first.")
        
        return predicted_class, confidence, all_probabilities
    
    def get_model_info(self):
        """
        获取模型信息
        Returns:
            model_info: 模型信息字典
        """
        if self.model_type == 'pytorch':
            return {
                'type': 'PyTorch',
                'model': str(self.model.__class__.__name__),
                'model_name': self.model_name,
                'device': str(self.device),
                'path': self.model_path,
                'load_error': self.load_error
            }
        elif self.model_type == 'onnx' and ONNX_AVAILABLE:
            return {
                'type': 'ONNX',
                'input_shape': self.session.get_inputs()[0].shape,
                'output_shape': self.session.get_outputs()[0].shape,
                'path': self.model_path,
                'load_error': self.load_error
            }
        else:
            return {
                'type': 'No model loaded',
                'load_error': self.load_error
            }
    
    def load_latest_model(self, model_dir='models/trained'):
        """
        自动加载最新训练的最佳模型
        Args:
            model_dir: 模型目录
        Returns:
            bool: 是否成功加载模型
            str: 加载的模型路径
        """
        import os
        
        # 检查模型目录是否存在
        if not os.path.exists(model_dir):
            print(f"模型目录不存在: {model_dir}")
            return False, ""
        
        # 收集所有训练文件夹，按时间戳排序
        train_folders = []
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            if os.path.isdir(item_path) and item.startswith('train_'):
                train_folders.append(item_path)
        
        if not train_folders:
            print("没有找到训练文件夹")
            return False, ""
        
        # 按文件夹名称排序（时间戳排序），最新的在前
        train_folders.sort(reverse=True)
        
        # 遍历文件夹，找到最佳模型
        latest_best_model = None
        highest_accuracy = 0.0
        best_model_name = None
        
        for folder in train_folders:
            # 在当前文件夹中查找所有最佳模型
            best_models_in_folder = []
            for file in os.listdir(folder):
                if file.startswith('best_model_') and file.endswith('.pth'):
                    best_models_in_folder.append(os.path.join(folder, file))
            
            if best_models_in_folder:
                # 找到当前文件夹中准确率最高的最佳模型
                folder_best_model = None
                folder_highest_accuracy = 0.0
                folder_model_name = 'enhanced'  # 默认使用enhanced模型架构
                
                for model_path in best_models_in_folder:
                    # 从文件名中提取准确率和架构信息
                    model_name = os.path.basename(model_path)
                    try:
                        # 解析文件名格式: best_model_96.32_20260112_155302.pth
                        parts = model_name.split('_')
                        accuracy_str = parts[2]
                        accuracy = float(accuracy_str)
                        
                        # 检查文件名中的架构信息
                        model_arch = None
                        if 'tiny' in model_name.lower():
                            model_arch = 'tiny'
                        elif 'enhanced' in model_name.lower():
                            model_arch = 'enhanced'
                        elif 'lightweight' in model_name.lower():
                            model_arch = 'lightweight'
                        elif 'advanced' in model_name.lower():
                            model_arch = 'advanced'
                        elif 'stm32' in model_name.lower():
                            model_arch = 'stm32'
                        
                        if accuracy > folder_highest_accuracy:
                            folder_highest_accuracy = accuracy
                            folder_best_model = model_path
                            if model_arch:
                                folder_model_name = model_arch
                    except:
                        pass
                
                if folder_best_model and folder_highest_accuracy > highest_accuracy:
                    highest_accuracy = folder_highest_accuracy
                    latest_best_model = folder_best_model
                    best_model_name = folder_model_name
                    
                    # 如果文件名中没有架构信息，尝试根据文件夹特征推断
                    if not best_model_name:
                        # 检查文件夹中是否有其他模型文件，推断架构
                        folder_files = os.listdir(folder)
                        if any('tiny' in f.lower() for f in folder_files):
                            best_model_name = 'tiny'
                        elif any('enhanced' in f.lower() for f in folder_files):
                            best_model_name = 'enhanced'
                        elif any('lightweight' in f.lower() for f in folder_files):
                            best_model_name = 'lightweight'
                        elif any('advanced' in f.lower() for f in folder_files):
                            best_model_name = 'advanced'
                        elif any('stm32' in f.lower() for f in folder_files):
                            best_model_name = 'stm32'
                        else:
                            best_model_name = 'enhanced'  # 默认使用enhanced
        
        if not latest_best_model:
            print("没有找到最佳模型")
            return False, ""
        
        print(f"尝试加载最新最佳模型: {latest_best_model}")
        print(f"最佳准确率: {highest_accuracy:.2f}%")
        print(f"推断的模型架构: {best_model_name}")
        
        try:
            # 尝试按顺序加载不同模型类型，优先使用推断的架构
            model_architectures = [best_model_name, 'tiny', 'lightweight', 'enhanced', 'advanced', 'stm32']
            
            for arch in model_architectures:
                try:
                    print(f"尝试使用 {arch} 架构加载...")
                    success = self.load_pytorch_model(latest_best_model, arch)
                    if success:
                        print(f"成功加载最新最佳模型: {latest_best_model}")
                        print(f"使用架构: {arch}")
                        return True, latest_best_model
                except Exception as e:
                    print(f"使用 {arch} 架构加载失败: {str(e)}")
                    continue
            
            # 如果所有架构都失败，抛出异常
            raise Exception("无法使用任何支持的架构加载模型")
            
        except Exception as e:
            error_msg = f"加载最新最佳模型失败: {str(e)}"
            self.load_error = error_msg
            print(error_msg)
            return False, ""